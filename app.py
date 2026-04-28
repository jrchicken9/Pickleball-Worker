from __future__ import annotations

import tempfile
import time
from pathlib import Path

import streamlit as st

from src.worker import (
    DEFAULT_PICKLEBALL_COURT_MODEL_PATH,
    process_video_for_court_detection,
    process_video_for_people,
    process_video_for_people_pose_movement,
)

DETECTION_MODELS = {
    "Nano (fastest)": "yolov8n.pt",
    "Small (better far detection)": "yolov8s.pt",
    "Medium (best of these, slower)": "yolov8m.pt",
}

POSE_MODELS = {
    "Nano Pose (fastest)": "yolov8n-pose.pt",
    "Small Pose (better far detection)": "yolov8s-pose.pt",
    "Medium Pose (best of these, slower)": "yolov8m-pose.pt",
}

INFERENCE_SIZE_OPTIONS = [640, 960, 1280]
DEVICE_OPTIONS = {
    "Auto (best available)": "auto",
    "CPU": "cpu",
    "Apple Silicon GPU (MPS)": "mps",
    "NVIDIA GPU (CUDA)": "cuda",
}


def _format_seconds(total_seconds: float) -> str:
    seconds = max(0, int(total_seconds))
    minutes, sec = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours:d}:{minutes:02d}:{sec:02d}"
    return f"{minutes:02d}:{sec:02d}"


def _init_state() -> None:
    if "result_video_bytes" not in st.session_state:
        st.session_state.result_video_bytes = None
    if "result_filename" not in st.session_state:
        st.session_state.result_filename = None
    if "result_stats" not in st.session_state:
        st.session_state.result_stats = None
    if "result_model_name" not in st.session_state:
        st.session_state.result_model_name = None
    if "result_inference_size" not in st.session_state:
        st.session_state.result_inference_size = None
    if "result_device" not in st.session_state:
        st.session_state.result_device = None


st.set_page_config(page_title="Human Detection Worker Test", layout="centered")
st.title("Human Detection Worker Test")
st.write(
    "Upload a video, run detection, and preview the processed output."
)
_init_state()

mode = st.selectbox(
    "Detection mode",
    options=[
        "Person boxes (fast, simple)",
        "Pose + moving joints (keypoints)",
        "Pickleball court (provider test)",
    ],
    index=1,
)
uploaded = st.file_uploader("Upload a video", type=["mp4", "mov", "avi", "mkv"])

if mode == "Pickleball court (provider test)":
    confidence = st.slider("Confidence threshold", 0.05, 0.95, 0.20, 0.05)
else:
    confidence = st.slider("Confidence threshold", 0.05, 0.95, 0.25, 0.05)

show_advanced = st.toggle("Show advanced settings", value=False)

# Sensible defaults keep the basic UI clean.
movement_threshold = 8.0
keypoint_threshold = 0.30
temporal_smoothing = True
hold_frames = 2
court_provider = "YOLO local"
court_class_name = "court skeleton"
roboflow_model_id = "liberin-technologies/pickleball-vision/6"
roboflow_api_key = ""
inference_size = 1280
selected_device = "auto"
selected_model_name = ""

if mode == "Pose + moving joints (keypoints)":
    selected_model_label = st.selectbox(
        "Pose model size",
        options=list(POSE_MODELS.keys()),
        index=1,
    )
    selected_model_name = POSE_MODELS[selected_model_label]
elif mode == "Pickleball court (provider test)":
    selected_model_label = st.selectbox(
        "Local court model file (for YOLO local)",
        options=[DEFAULT_PICKLEBALL_COURT_MODEL_PATH],
        index=0,
    )
    selected_model_name = selected_model_label
else:
    selected_model_label = st.selectbox(
        "Detection model size",
        options=list(DETECTION_MODELS.keys()),
        index=1,
    )
    selected_model_name = DETECTION_MODELS[selected_model_label]

if show_advanced:
    with st.expander("Advanced settings", expanded=True):
        inference_size = st.selectbox(
            "Inference size",
            options=INFERENCE_SIZE_OPTIONS,
            index=2,
            help="Larger sizes can improve far-point detection but run slower.",
        )
        selected_device_label = st.selectbox(
            "Inference device",
            options=list(DEVICE_OPTIONS.keys()),
            index=0,
            help="Auto picks the fastest available device.",
        )
        selected_device = DEVICE_OPTIONS[selected_device_label]

        if mode == "Person boxes (fast, simple)":
            temporal_smoothing = st.toggle(
                "Temporal smoothing (hold boxes briefly)",
                value=True,
            )
            hold_frames = st.slider(
                "Hold missing detections for N frames",
                0,
                5,
                2,
                1,
                disabled=not temporal_smoothing,
            )

        if mode == "Pose + moving joints (keypoints)":
            movement_threshold = st.slider(
                "Movement threshold (pixels)",
                1.0,
                30.0,
                8.0,
                1.0,
            )
            keypoint_threshold = st.slider(
                "Keypoint confidence threshold",
                0.05,
                0.95,
                0.30,
                0.05,
            )

        if mode == "Pickleball court (provider test)":
            court_provider = st.selectbox(
                "Court provider",
                options=["YOLO local", "Roboflow API"],
                index=0,
            )
            court_class_name = st.text_input(
                "Target class name",
                value="court skeleton",
                help="Set to model class (default for local GitHub model). Leave blank to keep all classes.",
            )
            if court_provider == "Roboflow API":
                roboflow_model_id = st.text_input(
                    "Roboflow model ID",
                    value="liberin-technologies/pickleball-vision/6",
                )
                roboflow_api_key = st.text_input(
                    "Roboflow API key",
                    value="",
                    type="password",
                )

if uploaded is not None:
    st.subheader("Original video")
    st.video(uploaded, format="video/mp4")
    if st.button("Run Detection", type="primary"):
        progress_bar = st.progress(0.0, text="Starting detection...")
        progress_text = st.empty()
        eta_text = st.empty()
        run_succeeded = False
        start_time = time.perf_counter()

        def on_progress(progress: float, processed: int, total: int) -> None:
            percent = int(progress * 100)
            progress_bar.progress(progress, text=f"Processing... {percent}%")
            progress_text.caption(f"Processed {processed}/{total} frames")
            if processed > 0:
                elapsed = time.perf_counter() - start_time
                avg_seconds_per_frame = elapsed / processed
                remaining_frames = max(0, total - processed)
                remaining_seconds = avg_seconds_per_frame * remaining_frames
                eta_text.caption(f"Estimated time remaining: {_format_seconds(remaining_seconds)}")

        try:
            with st.spinner("Processing video... this can take a bit."):
                with tempfile.TemporaryDirectory() as tmp_dir:
                    temp_dir = Path(tmp_dir)
                    input_path = temp_dir / uploaded.name
                    output_basename = uploaded.name.rsplit(".", 1)[0]
                    if mode == "Pose + moving joints (keypoints)":
                        output_path = (
                            temp_dir / f"tagged_pose_{output_basename}.mp4"
                        )
                    else:
                        output_path = temp_dir / f"tagged_{output_basename}.mp4"

                    input_path.write_bytes(uploaded.getbuffer())
                    if mode == "Pose + moving joints (keypoints)":
                        stats = process_video_for_people_pose_movement(
                            input_path=input_path,
                            output_path=output_path,
                            confidence_threshold=confidence,
                            movement_threshold_pixels=movement_threshold,
                            keypoint_threshold=keypoint_threshold,
                            model_name=selected_model_name,
                            inference_size=inference_size,
                            progress_callback=on_progress,
                            device=selected_device,
                        )
                    elif mode == "Person boxes (fast, simple)":
                        stats = process_video_for_people(
                            input_path=input_path,
                            output_path=output_path,
                            confidence_threshold=confidence,
                            model_name=selected_model_name,
                            inference_size=inference_size,
                            progress_callback=on_progress,
                            enable_temporal_smoothing=temporal_smoothing,
                            max_hold_frames=hold_frames,
                            device=selected_device,
                        )
                    else:
                        stats = process_video_for_court_detection(
                            input_path=input_path,
                            output_path=output_path,
                            confidence_threshold=confidence,
                            provider=(
                                "local"
                                if court_provider == "YOLO local"
                                else "roboflow"
                            ),
                            local_model_name=selected_model_name,
                            inference_size=inference_size,
                            device=selected_device,
                            roboflow_model_id=roboflow_model_id,
                            roboflow_api_key=roboflow_api_key,
                            target_class_name=court_class_name,
                            progress_callback=on_progress,
                            # Keep the latest/most accurate court pipeline fixed.
                            enable_stabilization=True,
                            reprojection_error_threshold=35.0,
                            smoothing_alpha=0.55,
                            hold_last_good_frames=2,
                            static_camera_lock=True,
                            lock_after_n_good_frames=10,
                            use_global_court_fit=True,
                        )
                    output_bytes = output_path.read_bytes()
                    output_filename = Path(stats.output_path).name
                    if mode == "Pickleball court (provider test)":
                        if court_provider == "YOLO local":
                            run_model_name = selected_model_name
                        else:
                            run_model_name = roboflow_model_id
                    else:
                        run_model_name = selected_model_name
                    run_succeeded = True
        except Exception as e:
            st.error(f"Detection failed: {e}")
            raise
        finally:
            if run_succeeded:
                progress_bar.progress(1.0, text="Processing... 100%")
                progress_text.caption("Processed video complete.")
                eta_text.caption("Estimated time remaining: 00:00")

        st.session_state.result_video_bytes = output_bytes
        st.session_state.result_filename = output_filename
        st.session_state.result_stats = stats
        st.session_state.result_model_name = run_model_name
        st.session_state.result_inference_size = inference_size
        st.session_state.result_device = selected_device

if st.session_state.result_video_bytes is not None:
    stats = st.session_state.result_stats
    st.success("Detection complete.")
    st.caption(
        "Last run used model "
        f"`{st.session_state.result_model_name}` at inference size "
        f"`{st.session_state.result_inference_size}` on device "
        f"`{st.session_state.result_device}`."
    )
    st.write(f"Frames processed: {stats.total_frames}")
    if hasattr(stats, "total_person_instances"):
        st.write(f"Frames with people: {stats.frames_with_people}")
        st.write(f"Total person instances: {stats.total_person_instances}")
        st.write(
            f"Frames with moving joints: {stats.frames_with_moving_joints}"
        )
        st.write(f"Trusted pose instances: {stats.trusted_pose_instances}")
        st.write(f"Average pose quality: {stats.average_pose_quality:.2f}")
        st.write(f"Average speed: {stats.average_speed_px_s:.1f} px/s")
        st.write(f"Peak speed: {stats.peak_speed_px_s:.1f} px/s")
    elif hasattr(stats, "total_court_detections"):
        st.write(f"Frames with court: {stats.frames_with_court}")
        st.write(f"Total court detections: {stats.total_court_detections}")
        st.write(f"Provider: {stats.provider}")
    else:
        st.write(f"Frames with people: {stats.frames_with_people}")
        st.write(f"Total people detections: {stats.total_people_detections}")

    show_download = st.toggle("Show download button", value=False)

    st.subheader("Tagged video preview")
    st.video(st.session_state.result_video_bytes, format="video/mp4")

    if show_download:
        st.download_button(
            "Download tagged video",
            data=st.session_state.result_video_bytes,
            file_name=st.session_state.result_filename,
            mime="video/mp4",
        )
