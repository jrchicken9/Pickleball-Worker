from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable
import base64

import cv2
import numpy as np
import requests
from ultralytics import YOLO


PERSON_CLASS_ID = 0
POSE_MODEL_NAME = "yolov8n-pose.pt"
DETECTION_MODEL_NAME = "yolov8n.pt"
DEFAULT_PICKLEBALL_COURT_MODEL_PATH = "models/court_detection.pt"
DEFAULT_PICKLEBALL_COURT_MODEL_URL = (
    "https://raw.githubusercontent.com/kpp91302/"
    "Pickleball-Analytics/main/models/court_detection.pt"
)
_MODEL_CACHE: dict[tuple[str, str], YOLO] = {}
DEFAULT_COURT_TEMPLATE_12 = np.array(
    [
        [0.0, 0.0],
        [200.0, 0.0],
        [400.0, 0.0],
        [0.0, 300.0],
        [200.0, 300.0],
        [400.0, 300.0],
        [0.0, 580.0],
        [200.0, 580.0],
        [400.0, 580.0],
        [0.0, 880.0],
        [200.0, 880.0],
        [400.0, 880.0],
    ],
    dtype=np.float32,
)

# COCO-17 keypoints skeleton (YOLOv8 pose uses COCO order).
#
# Keypoint indices:
# 0 nose, 1 left_eye, 2 right_eye, 3 left_ear, 4 right_ear,
# 5 left_shoulder, 6 right_shoulder, 7 left_elbow, 8 right_elbow,
# 9 left_wrist, 10 right_wrist, 11 left_hip, 12 right_hip,
# 13 left_knee, 14 right_knee, 15 left_ankle, 16 right_ankle
COCO_SKELETON_EDGES: list[tuple[int, int]] = [
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (0, 5),
    (0, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (11, 12),
    (5, 6),
]


def _resolve_device(device: str = "auto") -> str:
    """
    Resolve requested inference device.

    Supported inputs: auto, cpu, mps, cuda.
    """
    normalized = device.strip().lower()
    if normalized not in {"auto", "cpu", "mps", "cuda"}:
        raise ValueError(
            f"Unsupported device '{device}'. Use one of: auto, cpu, mps, cuda."
        )

    if normalized != "auto":
        if normalized == "cpu":
            return "cpu"
        try:
            import torch

            if normalized == "cuda":
                if not torch.cuda.is_available():
                    raise ValueError("CUDA device requested but CUDA is not available.")
                return "cuda:0"
            if normalized == "mps":
                if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
                    raise ValueError("MPS device requested but MPS is not available.")
                return "mps"
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError(f"Unable to validate device '{device}': {exc}") from exc

    try:
        import torch

        if torch.cuda.is_available():
            return "cuda:0"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass

    return "cpu"


def _get_cached_model(model_name: str, device: str = "auto") -> tuple[YOLO, str]:
    """Reuse loaded YOLO instances across runs to avoid warmup overhead."""
    resolved_device = _resolve_device(device)
    cache_key = (model_name, resolved_device)
    cached = _MODEL_CACHE.get(cache_key)
    if cached is not None:
        return cached, resolved_device

    model = YOLO(model_name)
    model.to(resolved_device)
    _MODEL_CACHE[cache_key] = model
    return model, resolved_device


@dataclass
class DetectionStats:
    total_frames: int
    frames_with_people: int
    total_people_detections: int
    output_path: str


def _create_video_writer(
    output_path: Path,
    fps: float,
    width: int,
    height: int,
) -> cv2.VideoWriter:
    """
    Create a writer with browser-friendly codec preference.

    We try H.264-compatible tags first for better Streamlit/browser playback.
    """
    codec_candidates = ["avc1", "H264", "mp4v"]
    for codec in codec_candidates:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        if writer.isOpened():
            return writer
        writer.release()

    raise ValueError(
        "Could not initialize a video writer with supported codecs "
        f"for output: {output_path}"
    )


def _ensure_local_model(local_model_name: str) -> str:
    """
    Ensure the requested local model exists.

    For the default pickleball court model, auto-download from GitHub once.
    """
    model_path = Path(local_model_name)
    if model_path.exists():
        return str(model_path)

    if local_model_name == DEFAULT_PICKLEBALL_COURT_MODEL_PATH:
        model_path.parent.mkdir(parents=True, exist_ok=True)
        resp = requests.get(DEFAULT_PICKLEBALL_COURT_MODEL_URL, timeout=60)
        resp.raise_for_status()
        model_path.write_bytes(resp.content)
        return str(model_path)

    raise ValueError(
        f"Local model not found: {local_model_name}. "
        "Provide an existing .pt path."
    )


def _bbox_iou(box_a: tuple[float, float, float, float], box_b: tuple[float, float, float, float]) -> float:
    """Compute intersection-over-union between two boxes (x1, y1, x2, y2)."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    if union <= 0.0:
        return 0.0
    return inter_area / union


def process_video_for_people(
    input_path: str | Path,
    output_path: str | Path,
    confidence_threshold: float = 0.25,
    model_name: str = DETECTION_MODEL_NAME,
    inference_size: int = 640,
    progress_callback: Callable[[float, int, int], None] | None = None,
    enable_temporal_smoothing: bool = True,
    max_hold_frames: int = 2,
    smoothing_iou_threshold: float = 0.3,
    device: str = "auto",
) -> DetectionStats:
    """Run person detection on each frame and render bounding boxes."""
    source = Path(input_path)
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {source}")
    total_frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    fps = capture.get(cv2.CAP_PROP_FPS) or 24.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = _create_video_writer(target, fps, width, height)

    model, resolved_device = _get_cached_model(model_name, device)
    predict_kwargs = {
        "conf": confidence_threshold,
        "classes": [PERSON_CLASS_ID],
        "imgsz": inference_size,
        "verbose": False,
        "device": resolved_device,
    }

    total_frames = 0
    frames_with_people = 0
    total_people_detections = 0
    active_tracks: list[dict] = []

    while True:
        ok, frame = capture.read()
        if not ok:
            break

        total_frames += 1
        result = model.predict(source=frame, **predict_kwargs)[0]

        boxes = result.boxes
        current_detections: list[dict] = []
        if boxes is not None and boxes.xyxy is not None:
            xyxy_all = boxes.xyxy.cpu().numpy()
            if boxes.conf is not None:
                confs_all = boxes.conf.cpu().numpy().reshape(-1)
            else:
                confs_all = np.ones((xyxy_all.shape[0],), dtype=np.float32)

            for i in range(xyxy_all.shape[0]):
                x1, y1, x2, y2 = xyxy_all[i]
                confidence = float(confs_all[i])
                current_detections.append(
                    {
                        "bbox": (float(x1), float(y1), float(x2), float(y2)),
                        "conf": confidence,
                    }
                )

        if enable_temporal_smoothing:
            unmatched_tracks = set(range(len(active_tracks)))
            unmatched_detections = set(range(len(current_detections)))

            # Greedy IoU matching between last known tracks and current detections.
            matches: list[tuple[int, int]] = []
            while unmatched_tracks and unmatched_detections:
                best_pair = None
                best_iou = -1.0
                for t_idx in unmatched_tracks:
                    for d_idx in unmatched_detections:
                        iou = _bbox_iou(
                            active_tracks[t_idx]["bbox"],
                            current_detections[d_idx]["bbox"],
                        )
                        if iou > best_iou:
                            best_iou = iou
                            best_pair = (t_idx, d_idx)

                if best_pair is None or best_iou < smoothing_iou_threshold:
                    break

                t_idx, d_idx = best_pair
                matches.append((t_idx, d_idx))
                unmatched_tracks.remove(t_idx)
                unmatched_detections.remove(d_idx)

            for t_idx, d_idx in matches:
                det = current_detections[d_idx]
                active_tracks[t_idx]["bbox"] = det["bbox"]
                active_tracks[t_idx]["conf"] = det["conf"]
                active_tracks[t_idx]["misses"] = 0

            for t_idx in unmatched_tracks:
                active_tracks[t_idx]["misses"] += 1

            for d_idx in unmatched_detections:
                det = current_detections[d_idx]
                active_tracks.append(
                    {"bbox": det["bbox"], "conf": det["conf"], "misses": 0}
                )

            active_tracks = [
                trk for trk in active_tracks if trk["misses"] <= max_hold_frames
            ]

            render_boxes = active_tracks
        else:
            render_boxes = [
                {"bbox": det["bbox"], "conf": det["conf"], "misses": 0}
                for det in current_detections
            ]

        people_in_frame = 0
        for tracked in render_boxes:
            x1, y1, x2, y2 = tracked["bbox"]
            confidence = tracked["conf"]
            is_held = tracked.get("misses", 0) > 0
            label = f"person {confidence:.2f}"
            if is_held:
                label += " (held)"

            color = (0, 255, 0) if not is_held else (0, 180, 255)
            cv2.rectangle(
                frame,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                color,
                2,
            )
            cv2.putText(
                frame,
                label,
                (int(x1), max(18, int(y1) - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )
            people_in_frame += 1

        if people_in_frame > 0:
            frames_with_people += 1
            total_people_detections += people_in_frame

        writer.write(frame)
        if progress_callback is not None and total_frame_count > 0:
            progress_callback(
                min(1.0, total_frames / total_frame_count),
                total_frames,
                total_frame_count,
            )

    capture.release()
    writer.release()

    return DetectionStats(
        total_frames=total_frames,
        frames_with_people=frames_with_people,
        total_people_detections=total_people_detections,
        output_path=str(target),
    )


@dataclass
class PoseStats:
    total_frames: int
    frames_with_people: int
    total_person_instances: int
    frames_with_moving_joints: int
    trusted_pose_instances: int
    average_pose_quality: float
    average_speed_px_s: float
    peak_speed_px_s: float
    output_path: str


@dataclass
class CourtStats:
    total_frames: int
    frames_with_court: int
    total_court_detections: int
    provider: str
    output_path: str


def _draw_pose_instance(
    frame: np.ndarray,
    kpt_xy: np.ndarray,  # (17,2)
    kpt_conf: np.ndarray,  # (17,)
    instance_conf: float,
    moving_mask: np.ndarray | None = None,  # (17,)
    keypoint_threshold: float = 0.3,
    extra_label: str = "",
    is_held: bool = False,
) -> None:
    """Draw pose keypoints + skeleton for one detected person."""
    moving_mask = moving_mask if moving_mask is not None else np.zeros(17, dtype=bool)

    # Bounding label anchor: nose if available; otherwise fallback to first confident joint.
    nose_idx = 0
    label_pt = None
    if kpt_conf[nose_idx] >= keypoint_threshold:
        label_pt = tuple(kpt_xy[nose_idx].astype(int).tolist())
    else:
        for j in range(17):
            if kpt_conf[j] >= keypoint_threshold:
                label_pt = tuple(kpt_xy[j].astype(int).tolist())
                break

    # Draw skeleton edges first.
    for a, b in COCO_SKELETON_EDGES:
        if kpt_conf[a] < keypoint_threshold or kpt_conf[b] < keypoint_threshold:
            continue

        if is_held:
            color = (180, 180, 180)
        else:
            color = (255, 0, 0) if (moving_mask[a] or moving_mask[b]) else (0, 200, 255)
        ax, ay = kpt_xy[a].astype(int).tolist()
        bx, by = kpt_xy[b].astype(int).tolist()
        cv2.line(frame, (ax, ay), (bx, by), color, 2)

    # Draw keypoint circles.
    for j in range(17):
        if kpt_conf[j] < keypoint_threshold:
            continue
        x, y = kpt_xy[j].astype(int).tolist()
        if is_held:
            color = (180, 180, 180)
        else:
            color = (0, 255, 0) if not moving_mask[j] else (0, 0, 255)
        cv2.circle(frame, (x, y), 4, color, -1, lineType=cv2.LINE_AA)

    if label_pt is not None:
        label = f"person-pose {instance_conf:.2f}"
        if extra_label:
            label = f"{label} {extra_label}"
        if is_held:
            label = f"{label} (held)"
        x, y = label_pt
        cv2.putText(
            frame,
            label,
            (x, max(18, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )


def _safe_center_from_indices(
    kpt_xy: np.ndarray,
    kpt_conf: np.ndarray,
    indices: list[int],
    keypoint_threshold: float,
) -> tuple[float, float] | None:
    points = []
    for idx in indices:
        if kpt_conf[idx] >= keypoint_threshold:
            points.append(kpt_xy[idx])
    if not points:
        return None
    arr = np.stack(points, axis=0)
    return float(arr[:, 0].mean()), float(arr[:, 1].mean())


def _direction_from_delta(dx: float, dy: float) -> str:
    if abs(dx) < 1e-5 and abs(dy) < 1e-5:
        return "still"
    angle = np.degrees(np.arctan2(-dy, dx))
    if -22.5 <= angle < 22.5:
        return "right"
    if 22.5 <= angle < 67.5:
        return "up-right"
    if 67.5 <= angle < 112.5:
        return "up"
    if 112.5 <= angle < 157.5:
        return "up-left"
    if angle >= 157.5 or angle < -157.5:
        return "left"
    if -157.5 <= angle < -112.5:
        return "down-left"
    if -112.5 <= angle < -67.5:
        return "down"
    return "down-right"


def process_video_for_people_pose_movement(
    input_path: str | Path,
    output_path: str | Path,
    confidence_threshold: float = 0.25,
    movement_threshold_pixels: float = 8.0,
    keypoint_threshold: float = 0.3,
    model_name: str = POSE_MODEL_NAME,
    inference_size: int = 640,
    progress_callback: Callable[[float, int, int], None] | None = None,
    pose_quality_min_avg_conf: float = 0.35,
    pose_quality_min_visible_kpts: int = 4,
    occlusion_hold_frames: int = 2,
    occlusion_match_distance_px: float = 120.0,
    device: str = "auto",
) -> PoseStats:
    """
    Run pose estimation on each frame and highlight moving joints.

    Movement is approximated by frame-to-frame displacement of matched poses.
    """
    source = Path(input_path)
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {source}")
    total_frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    fps = capture.get(cv2.CAP_PROP_FPS) or 24.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = _create_video_writer(target, fps, width, height)

    model, resolved_device = _get_cached_model(model_name, device)
    predict_kwargs = {
        "conf": confidence_threshold,
        "imgsz": inference_size,
        "verbose": False,
        "device": resolved_device,
    }

    total_frames = 0
    frames_with_people = 0
    total_person_instances = 0
    frames_with_moving_joints = 0
    trusted_pose_instances = 0
    pose_quality_sum = 0.0
    speed_samples_sum = 0.0
    speed_samples_count = 0
    peak_speed_px_s = 0.0

    # Active tracks for light occlusion robustness and motion estimates.
    active_tracks: list[dict] = []

    while True:
        ok, frame = capture.read()
        if not ok:
            break

        total_frames += 1

        result = model.predict(source=frame, **predict_kwargs)[0]

        # Ultralytics pose results expose `keypoints` and also `boxes`.
        curr_instances: list[dict] = []
        people_in_frame = 0

        boxes = getattr(result, "boxes", None)
        keypoints = getattr(result, "keypoints", None)
        if boxes is not None and keypoints is not None and keypoints.xy is not None:
            # keypoints.xy: (n, 17, 2), keypoints.conf: (n, 17)
            kpt_xy_all = keypoints.xy.cpu().numpy()

            if keypoints.conf is not None:
                kpt_conf_all = keypoints.conf.cpu().numpy()
            else:
                kpt_conf_all = np.ones((kpt_xy_all.shape[0], kpt_xy_all.shape[1]), dtype=np.float32)

            if boxes.xyxy is not None:
                boxes_xyxy_all = boxes.xyxy.cpu().numpy()
            else:
                boxes_xyxy_all = np.zeros((kpt_xy_all.shape[0], 4), dtype=np.float32)

            if boxes.conf is not None:
                confs_all = boxes.conf.cpu().numpy().reshape(-1)
            else:
                confs_all = np.ones((kpt_xy_all.shape[0],), dtype=np.float32)

            for i in range(kpt_xy_all.shape[0]):
                x1, y1, x2, y2 = boxes_xyxy_all[i]
                cx = float((x1 + x2) / 2.0)
                cy = float((y1 + y2) / 2.0)
                kpt_xy = kpt_xy_all[i]
                kpt_conf = kpt_conf_all[i]

                visible_mask = kpt_conf >= keypoint_threshold
                visible_count = int(np.count_nonzero(visible_mask))
                pose_quality = (
                    float(kpt_conf[visible_mask].mean()) if visible_count > 0 else 0.0
                )
                trusted_pose = (
                    visible_count >= pose_quality_min_visible_kpts
                    and pose_quality >= pose_quality_min_avg_conf
                )

                core_center = _safe_center_from_indices(
                    kpt_xy=kpt_xy,
                    kpt_conf=kpt_conf,
                    indices=[5, 6, 11, 12],  # shoulders + hips
                    keypoint_threshold=keypoint_threshold,
                )
                motion_center = (
                    core_center if core_center is not None else (cx, cy)
                )

                curr_instances.append(
                    {
                        "cx": cx,
                        "cy": cy,
                        "kpt_xy": kpt_xy,
                        "kpt_conf": kpt_conf,
                        "instance_conf": float(confs_all[i]),
                        "match_center": motion_center,
                        "motion_center": motion_center,
                        "pose_quality": pose_quality,
                        "visible_count": visible_count,
                        "trusted_pose": trusted_pose,
                    }
                )
                people_in_frame += 1

        if people_in_frame > 0:
            frames_with_people += 1
            total_person_instances += people_in_frame

        # Match current instances to active tracks, preferring torso/hip center.
        unmatched_tracks = set(range(len(active_tracks)))
        unmatched_curr = set(range(len(curr_instances)))
        matches: list[tuple[int, int]] = []

        while unmatched_tracks and unmatched_curr:
            best_pair = None
            best_dist = float("inf")
            for t_idx in unmatched_tracks:
                track = active_tracks[t_idx]
                tx, ty = track["match_center"]
                for c_idx in unmatched_curr:
                    curr = curr_instances[c_idx]
                    cx, cy = curr["match_center"]
                    dist = ((tx - cx) ** 2 + (ty - cy) ** 2) ** 0.5
                    if dist < best_dist:
                        best_dist = dist
                        best_pair = (t_idx, c_idx)

            if best_pair is None or best_dist > occlusion_match_distance_px:
                break

            t_idx, c_idx = best_pair
            matches.append((t_idx, c_idx))
            unmatched_tracks.remove(t_idx)
            unmatched_curr.remove(c_idx)

        any_moving = False

        for t_idx, c_idx in matches:
            track = active_tracks[t_idx]
            curr = curr_instances[c_idx]
            curr_xy = curr["kpt_xy"]
            curr_conf = curr["kpt_conf"]

            prev_xy = track["kpt_xy"]
            prev_conf = track["kpt_conf"]

            disp = np.linalg.norm(curr_xy - prev_xy, axis=-1)  # (17,)
            moving_mask = (
                (disp >= movement_threshold_pixels)
                & (curr_conf >= keypoint_threshold)
                & (prev_conf >= keypoint_threshold)
            )
            if moving_mask.any():
                any_moving = True

            prev_center = track["motion_center"]
            curr_center = curr["motion_center"]
            dx = curr_center[0] - prev_center[0]
            dy = curr_center[1] - prev_center[1]
            speed_px_s = (((dx * dx) + (dy * dy)) ** 0.5) * fps
            direction = _direction_from_delta(dx, dy)

            if curr["trusted_pose"]:
                speed_samples_sum += speed_px_s
                speed_samples_count += 1
                peak_speed_px_s = max(peak_speed_px_s, speed_px_s)

            track.update(
                {
                    "cx": curr["cx"],
                    "cy": curr["cy"],
                    "match_center": curr["match_center"],
                    "motion_center": curr["motion_center"],
                    "kpt_xy": curr_xy,
                    "kpt_conf": curr_conf,
                    "instance_conf": curr["instance_conf"],
                    "misses": 0,
                    "moving_mask": moving_mask,
                    "pose_quality": curr["pose_quality"],
                    "visible_count": curr["visible_count"],
                    "trusted_pose": curr["trusted_pose"],
                    "speed_px_s": speed_px_s,
                    "direction": direction,
                }
            )

        for t_idx in unmatched_tracks:
            active_tracks[t_idx]["misses"] += 1
            active_tracks[t_idx]["moving_mask"] = np.zeros(17, dtype=bool)

        for c_idx in unmatched_curr:
            curr = curr_instances[c_idx]
            active_tracks.append(
                {
                    "cx": curr["cx"],
                    "cy": curr["cy"],
                    "match_center": curr["match_center"],
                    "motion_center": curr["motion_center"],
                    "kpt_xy": curr["kpt_xy"],
                    "kpt_conf": curr["kpt_conf"],
                    "instance_conf": curr["instance_conf"],
                    "misses": 0,
                    "moving_mask": np.zeros(17, dtype=bool),
                    "pose_quality": curr["pose_quality"],
                    "visible_count": curr["visible_count"],
                    "trusted_pose": curr["trusted_pose"],
                    "speed_px_s": 0.0,
                    "direction": "still",
                }
            )

        active_tracks = [
            trk for trk in active_tracks if trk["misses"] <= occlusion_hold_frames
        ]

        # Aggregate quality stats from raw detections (not held tracks).
        for curr in curr_instances:
            pose_quality_sum += curr["pose_quality"]
            if curr["trusted_pose"]:
                trusted_pose_instances += 1

        # Draw active tracks (including briefly held ones).
        any_moving = False
        for track in active_tracks:
            if track["moving_mask"].any():
                any_moving = True

            quality_tag = (
                "trusted"
                if track["trusted_pose"]
                else "lowQ"
            )
            extra = (
                f"q={track['pose_quality']:.2f} "
                f"vis={track['visible_count']} "
                f"spd={track['speed_px_s']:.1f}px/s "
                f"dir={track['direction']} "
                f"{quality_tag}"
            )
            _draw_pose_instance(
                frame=frame,
                kpt_xy=track["kpt_xy"],
                kpt_conf=track["kpt_conf"],
                instance_conf=track["instance_conf"],
                moving_mask=track["moving_mask"],
                keypoint_threshold=keypoint_threshold,
                extra_label=extra,
                is_held=track["misses"] > 0,
            )

        if any_moving:
            frames_with_moving_joints += 1

        writer.write(frame)
        if progress_callback is not None and total_frame_count > 0:
            progress_callback(
                min(1.0, total_frames / total_frame_count),
                total_frames,
                total_frame_count,
            )

    capture.release()
    writer.release()

    average_pose_quality = (
        pose_quality_sum / total_person_instances if total_person_instances > 0 else 0.0
    )
    average_speed_px_s = (
        speed_samples_sum / speed_samples_count if speed_samples_count > 0 else 0.0
    )

    return PoseStats(
        total_frames=total_frames,
        frames_with_people=frames_with_people,
        total_person_instances=total_person_instances,
        frames_with_moving_joints=frames_with_moving_joints,
        trusted_pose_instances=trusted_pose_instances,
        average_pose_quality=average_pose_quality,
        average_speed_px_s=average_speed_px_s,
        peak_speed_px_s=peak_speed_px_s,
        output_path=str(target),
    )


def _roboflow_infer_frame(
    frame_bgr: np.ndarray,
    model_id: str,
    api_key: str,
    confidence_threshold: float,
) -> list[dict]:
    ok, encoded = cv2.imencode(".jpg", frame_bgr)
    if not ok:
        return []
    img_b64 = base64.b64encode(encoded.tobytes()).decode("utf-8")
    url = (
        f"https://detect.roboflow.com/{model_id}"
        f"?api_key={api_key}&confidence={int(confidence_threshold * 100)}"
    )
    resp = requests.post(
        url,
        data=img_b64,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=30,
    )
    resp.raise_for_status()
    payload = resp.json()
    preds = payload.get("predictions", [])
    parsed: list[dict] = []
    for pred in preds:
        x = float(pred.get("x", 0.0))
        y = float(pred.get("y", 0.0))
        w = float(pred.get("width", 0.0))
        h = float(pred.get("height", 0.0))
        x1 = x - (w / 2.0)
        y1 = y - (h / 2.0)
        x2 = x + (w / 2.0)
        y2 = y + (h / 2.0)
        parsed.append(
            {
                "bbox": (x1, y1, x2, y2),
                "class": str(pred.get("class", "court")),
                "conf": float(pred.get("confidence", 0.0)),
            }
        )
    return parsed


def _draw_court_detection_overlay(frame: np.ndarray, detection: dict) -> None:
    color = (255, 120, 0)
    conf = float(detection.get("conf", 0.0))
    cls_name = str(detection.get("class", "court"))
    det_type = str(detection.get("type", "bbox"))
    is_held = bool(detection.get("is_held", False))
    is_stabilized = bool(detection.get("is_stabilized", False))
    is_locked = bool(detection.get("is_locked", False))

    if det_type == "polygon":
        area_points = detection.get("area_points", [])
        points = area_points if len(area_points) >= 3 else detection.get("points", [])
        if len(points) < 3:
            return
        poly = np.array(points, dtype=np.int32).reshape(-1, 1, 2)

        # Fill court area with transparent overlay for clearer region visualization.
        overlay = frame.copy()
        cv2.fillPoly(overlay, [poly], color)
        cv2.addWeighted(overlay, 0.22, frame, 0.78, 0.0, frame)

        cv2.polylines(frame, [poly], isClosed=True, color=color, thickness=3)
        for x, y in points:
            cv2.circle(frame, (int(x), int(y)), 3, color, -1, lineType=cv2.LINE_AA)

        anchor_x = int(min(p[0] for p in points))
        anchor_y = int(min(p[1] for p in points))
        if is_locked:
            suffix = " locked"
        elif is_held:
            suffix = " held"
        elif is_stabilized:
            suffix = " stabilized"
        else:
            suffix = ""
        label = f"{cls_name} area {conf:.2f}{suffix}"
        cv2.putText(
            frame,
            label,
            (anchor_x, max(18, anchor_y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )
        return

    x1, y1, x2, y2 = detection["bbox"]
    if is_locked:
        suffix = " locked"
    elif is_held:
        suffix = " held"
    elif is_stabilized:
        suffix = " stabilized"
    else:
        suffix = ""
    label = f"{cls_name} {conf:.2f}{suffix}"
    cv2.rectangle(
        frame,
        (int(x1), int(y1)),
        (int(x2), int(y2)),
        color,
        2,
    )
    cv2.putText(
        frame,
        label,
        (int(x1), max(18, int(y1) - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
        cv2.LINE_AA,
    )


def _court_area_polygon_from_keypoints(points: np.ndarray) -> list[tuple[float, float]]:
    """Build a 4-corner court polygon from keypoints, robust to point ordering."""
    if points.shape[0] < 4:
        return [(float(x), float(y)) for x, y in points]

    if points.shape[0] >= 12:
        # Dedicated pickleball court model uses 12 ordered keypoints.
        # These indices correspond to the outer court corners.
        corner_indices = [0, 2, 11, 9]  # tl, tr, br, bl
        return [(float(points[i, 0]), float(points[i, 1])) for i in corner_indices]

    hull = cv2.convexHull(points.astype(np.float32)).reshape(-1, 2)
    work_pts = hull if hull.shape[0] >= 4 else points

    # Corner extraction from arbitrary point cloud.
    s = work_pts[:, 0] + work_pts[:, 1]
    d = work_pts[:, 0] - work_pts[:, 1]
    tl = work_pts[np.argmin(s)]
    br = work_pts[np.argmax(s)]
    tr = work_pts[np.argmax(d)]
    bl = work_pts[np.argmin(d)]

    quad = np.array([tl, tr, br, bl], dtype=np.float32)
    return [(float(x), float(y)) for x, y in quad]


def _homography_reprojection_error(
    src_points: np.ndarray,
    dst_points: np.ndarray,
) -> float | None:
    """Estimate homography fit quality via mean reprojection error."""
    if src_points.shape[0] < 4 or dst_points.shape[0] < 4:
        return None
    H, _ = cv2.findHomography(src_points, dst_points, method=cv2.RANSAC)
    if H is None:
        return None
    projected = cv2.perspectiveTransform(src_points.reshape(-1, 1, 2), H).reshape(-1, 2)
    error = np.linalg.norm(projected - dst_points, axis=1).mean()
    return float(error)


def _smooth_keypoints(
    prev_points: np.ndarray,
    curr_points: np.ndarray,
    alpha: float,
) -> np.ndarray:
    return (alpha * curr_points) + ((1.0 - alpha) * prev_points)


def _median_polygon(polygons: list[np.ndarray]) -> np.ndarray:
    stacked = np.stack(polygons, axis=0)  # (n, 4, 2)
    return np.median(stacked, axis=0).astype(np.float32)


def _court_white_line_mask(frame_bgr: np.ndarray) -> np.ndarray:
    """Binary mask of likely white court lines."""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    # White lines: high value, low saturation.
    mask = cv2.inRange(hsv, (0, 0, 165), (180, 70, 255))
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    return mask


def _edge_score(mask: np.ndarray, p1: np.ndarray, p2: np.ndarray, samples: int = 150) -> float:
    h, w = mask.shape[:2]
    xs = np.linspace(float(p1[0]), float(p2[0]), samples)
    ys = np.linspace(float(p1[1]), float(p2[1]), samples)
    x_int = np.clip(np.round(xs).astype(int), 0, w - 1)
    y_int = np.clip(np.round(ys).astype(int), 0, h - 1)
    vals = mask[y_int, x_int] / 255.0
    return float(vals.mean())


def _quad_line_score(mask: np.ndarray, quad: np.ndarray) -> float:
    # quad order: tl, tr, br, bl
    s = 0.0
    for a, b in [(0, 1), (1, 2), (2, 3), (3, 0)]:
        s += _edge_score(mask, quad[a], quad[b])
    return s / 4.0


def _refine_quad_to_lines(
    frame_bgr: np.ndarray,
    quad: np.ndarray,
    search_radius: int = 18,
    step: int = 2,
) -> np.ndarray:
    """
    Refine a court quad by maximizing overlap with bright line pixels.

    This runs once per video in global-fit mode.
    """
    if quad.shape != (4, 2):
        return quad

    mask = _court_white_line_mask(frame_bgr)
    refined = quad.astype(np.float32).copy()
    best_global = _quad_line_score(mask, refined)

    # Coordinate-descent over corners in local windows.
    for _ in range(2):
        improved_any = False
        for i in range(4):
            base = refined[i].copy()
            best_local_pt = base
            best_local_score = best_global
            for dx in range(-search_radius, search_radius + 1, step):
                for dy in range(-search_radius, search_radius + 1, step):
                    cand = refined.copy()
                    cand[i, 0] = base[0] + dx
                    cand[i, 1] = base[1] + dy
                    score = _quad_line_score(mask, cand)
                    if score > best_local_score:
                        best_local_score = score
                        best_local_pt = cand[i].copy()
            if best_local_score > best_global:
                refined[i] = best_local_pt
                best_global = best_local_score
                improved_any = True
        if not improved_any:
            break
    return refined


def _scale_quad_about_center(quad: np.ndarray, scale: float) -> np.ndarray:
    center = quad.mean(axis=0, keepdims=True)
    return center + (quad - center) * scale


def _calibrate_quad_scale_to_lines(
    frame_bgr: np.ndarray,
    quad: np.ndarray,
    min_scale: float = 0.90,
    max_scale: float = 1.15,
    step: float = 0.01,
) -> np.ndarray:
    """
    Tune global quad size against white-line evidence.

    This helps when keypoints are consistently slightly inset or outset.
    """
    if quad.shape != (4, 2):
        return quad
    mask = _court_white_line_mask(frame_bgr)
    best_quad = quad.copy()
    best_score = _quad_line_score(mask, best_quad)
    s = min_scale
    while s <= max_scale + 1e-9:
        cand = _scale_quad_about_center(quad, s)
        score = _quad_line_score(mask, cand)
        if score > best_score:
            best_score = score
            best_quad = cand
        s += step
    return best_quad


def _line_intersection(
    p1: np.ndarray,
    p2: np.ndarray,
    p3: np.ndarray,
    p4: np.ndarray,
) -> np.ndarray | None:
    """Intersection of infinite lines (p1,p2) and (p3,p4)."""
    x1, y1 = float(p1[0]), float(p1[1])
    x2, y2 = float(p2[0]), float(p2[1])
    x3, y3 = float(p3[0]), float(p3[1])
    x4, y4 = float(p4[0]), float(p4[1])
    den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(den) < 1e-6:
        return None
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / den
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / den
    return np.array([px, py], dtype=np.float32)


def _refine_quad_edges_to_lines(
    frame_bgr: np.ndarray,
    quad: np.ndarray,
    max_offset: int = 32,
    step: int = 2,
) -> np.ndarray:
    """
    Snap each quad edge independently to high-confidence white-line pixels.

    This usually fixes residual inset/outset errors that uniform scaling cannot.
    """
    if quad.shape != (4, 2):
        return quad

    mask = _court_white_line_mask(frame_bgr)
    q = quad.astype(np.float32).copy()  # tl,tr,br,bl (clockwise)
    edges = [(0, 1), (1, 2), (2, 3), (3, 0)]

    shifted_edges: list[tuple[np.ndarray, np.ndarray]] = []
    for a, b in edges:
        p1 = q[a].copy()
        p2 = q[b].copy()
        e = p2 - p1
        norm = np.linalg.norm(e)
        if norm < 1e-6:
            shifted_edges.append((p1, p2))
            continue
        # For clockwise polygon, outward normal is (dy, -dx).
        n = np.array([e[1], -e[0]], dtype=np.float32) / norm

        best_score = -1.0
        best_t = 0.0
        for t in range(-max_offset, max_offset + 1, step):
            s1 = p1 + n * float(t)
            s2 = p2 + n * float(t)
            score = _edge_score(mask, s1, s2)
            if score > best_score:
                best_score = score
                best_t = float(t)

        shifted_edges.append((p1 + n * best_t, p2 + n * best_t))

    # Rebuild corners as intersections of adjacent shifted edges.
    top = shifted_edges[0]
    right = shifted_edges[1]
    bottom = shifted_edges[2]
    left = shifted_edges[3]

    tl = _line_intersection(top[0], top[1], left[0], left[1])
    tr = _line_intersection(top[0], top[1], right[0], right[1])
    br = _line_intersection(right[0], right[1], bottom[0], bottom[1])
    bl = _line_intersection(bottom[0], bottom[1], left[0], left[1])

    if tl is None or tr is None or br is None or bl is None:
        return q

    refined = np.stack([tl, tr, br, bl], axis=0).astype(np.float32)
    return refined


def _class_matches(target_class_norm: str, detected_class: str) -> bool:
    """
    Flexible class matching for custom community models.

    Matches exact, substring, and token-overlap cases
    (e.g. target "court" vs detected "court skeleton").
    """
    if not target_class_norm:
        return True
    detected_norm = detected_class.strip().lower()
    if not detected_norm:
        return False
    if detected_norm == target_class_norm:
        return True
    if target_class_norm in detected_norm or detected_norm in target_class_norm:
        return True
    target_tokens = set(target_class_norm.split())
    detected_tokens = set(detected_norm.split())
    return bool(target_tokens & detected_tokens)


def process_video_for_court_detection(
    input_path: str | Path,
    output_path: str | Path,
    confidence_threshold: float = 0.25,
    provider: str = "local",
    local_model_name: str = DEFAULT_PICKLEBALL_COURT_MODEL_PATH,
    inference_size: int = 1280,
    device: str = "auto",
    roboflow_model_id: str = "liberin-technologies/pickleball-vision/6",
    roboflow_api_key: str = "",
    target_class_name: str = "",
    progress_callback: Callable[[float, int, int], None] | None = None,
    enable_stabilization: bool = True,
    reprojection_error_threshold: float = 35.0,
    smoothing_alpha: float = 0.55,
    hold_last_good_frames: int = 2,
    static_camera_lock: bool = True,
    lock_after_n_good_frames: int = 10,
    use_global_court_fit: bool = True,
) -> CourtStats:
    source = Path(input_path)
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {source}")
    total_frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = capture.get(cv2.CAP_PROP_FPS) or 24.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()

    normalized_provider = provider.strip().lower()
    if normalized_provider not in {"local", "roboflow"}:
        raise ValueError("provider must be 'local' or 'roboflow'")

    local_model = None
    local_predict_kwargs: dict | None = None
    if normalized_provider == "local":
        ensured_model_path = _ensure_local_model(local_model_name)
        local_model, resolved_device = _get_cached_model(ensured_model_path, device)
        local_predict_kwargs = {
            "conf": confidence_threshold,
            "imgsz": inference_size,
            "verbose": False,
            "device": resolved_device,
        }
    else:
        if not roboflow_api_key:
            raise ValueError(
                "Roboflow API key is required for provider='roboflow'."
            )

    total_frames = 0
    frames_with_court = 0
    total_court_detections = 0
    target_class_norm = target_class_name.strip().lower()
    last_good_keypoints: np.ndarray | None = None
    last_good_detection: dict | None = None
    missing_streak = 0
    locked_polygon: np.ndarray | None = None
    good_polygons_for_lock: list[np.ndarray] = []

    def detect_local_frame(frame_bgr: np.ndarray) -> list[dict]:
        assert local_model is not None and local_predict_kwargs is not None
        result = local_model.predict(source=frame_bgr, **local_predict_kwargs)[0]
        boxes = getattr(result, "boxes", None)
        keypoints = getattr(result, "keypoints", None)
        frame_detections: list[dict] = []

        if keypoints is not None and keypoints.xy is not None:
            kpt_xy_all = keypoints.xy.cpu().numpy()
            if boxes is not None and boxes.conf is not None:
                confs_all = boxes.conf.cpu().numpy().reshape(-1)
            else:
                confs_all = np.ones((kpt_xy_all.shape[0],), dtype=np.float32)
            if boxes is not None and boxes.cls is not None:
                class_ids = boxes.cls.cpu().numpy().astype(int)
            else:
                class_ids = np.zeros((kpt_xy_all.shape[0],), dtype=int)

            names_map = result.names if hasattr(result, "names") else {}
            for i in range(kpt_xy_all.shape[0]):
                cls_name = str(names_map.get(int(class_ids[i]), target_class_name))
                if not _class_matches(target_class_norm, cls_name):
                    continue
                pts = []
                for x, y in kpt_xy_all[i]:
                    if np.isfinite(x) and np.isfinite(y) and x > 0 and y > 0:
                        pts.append((float(x), float(y)))
                if len(pts) >= 3:
                    points_np = np.array(pts, dtype=np.float32)
                    frame_detections.append(
                        {
                            "type": "polygon",
                            "points": pts,
                            "area_points": _court_area_polygon_from_keypoints(points_np),
                            "class": cls_name,
                            "conf": float(confs_all[i]),
                        }
                    )

        if not frame_detections and boxes is not None and boxes.xyxy is not None:
            xyxy_all = boxes.xyxy.cpu().numpy()
            confs_all = (
                boxes.conf.cpu().numpy().reshape(-1)
                if boxes.conf is not None
                else np.ones((xyxy_all.shape[0],), dtype=np.float32)
            )
            class_ids = (
                boxes.cls.cpu().numpy().astype(int)
                if boxes.cls is not None
                else np.zeros((xyxy_all.shape[0],), dtype=int)
            )
            names_map = result.names if hasattr(result, "names") else {}
            for i in range(xyxy_all.shape[0]):
                cls_name = str(names_map.get(int(class_ids[i]), int(class_ids[i])))
                if not _class_matches(target_class_norm, cls_name):
                    continue
                x1, y1, x2, y2 = xyxy_all[i]
                frame_detections.append(
                    {
                        "type": "bbox",
                        "bbox": (float(x1), float(y1), float(x2), float(y2)),
                        "class": cls_name,
                        "conf": float(confs_all[i]),
                    }
                )
        return frame_detections

    final_global_detection: dict | None = None
    if normalized_provider == "local" and use_global_court_fit:
        pass1 = cv2.VideoCapture(str(source))
        if not pass1.isOpened():
            raise ValueError(f"Could not open video: {source}")
        candidate_polygons: list[np.ndarray] = []
        candidate_keypoints12: list[np.ndarray] = []
        confs: list[float] = []
        processed_pass1 = 0
        best_frame_for_refine: np.ndarray | None = None
        best_conf_for_refine = -1.0

        while True:
            ok, frame = pass1.read()
            if not ok:
                break
            processed_pass1 += 1
            frame_detections = detect_local_frame(frame)
            if frame_detections:
                best_det = max(frame_detections, key=lambda d: float(d.get("conf", 0.0)))
                if best_det.get("type") == "polygon":
                    points_np = np.array(best_det.get("points", []), dtype=np.float32)
                    if points_np.shape[0] >= 4:
                        if points_np.shape[0] >= 12:
                            dst = DEFAULT_COURT_TEMPLATE_12[: points_np.shape[0]]
                            err = _homography_reprojection_error(points_np, dst)
                            if err is None or err > reprojection_error_threshold:
                                continue
                        quad = np.array(best_det["area_points"], dtype=np.float32)
                        if quad.shape == (4, 2):
                            candidate_polygons.append(quad)
                            if points_np.shape[0] >= 12:
                                candidate_keypoints12.append(points_np[:12].copy())
                            det_conf = float(best_det.get("conf", 0.0))
                            confs.append(det_conf)
                            if det_conf > best_conf_for_refine:
                                best_conf_for_refine = det_conf
                                best_frame_for_refine = frame.copy()

            if progress_callback is not None and total_frame_count > 0:
                progress_callback(
                    0.5 * min(1.0, processed_pass1 / total_frame_count),
                    processed_pass1,
                    total_frame_count,
                )
        pass1.release()

        if candidate_polygons:
            if candidate_keypoints12:
                # Aggregate the full ordered keypoint set first for better geometry.
                stacked_kpts = np.stack(candidate_keypoints12, axis=0)  # (n,12,2)
                median_kpts = np.median(stacked_kpts, axis=0).astype(np.float32)
                coarse = np.array(
                    _court_area_polygon_from_keypoints(median_kpts), dtype=np.float32
                )
            else:
                coarse = _median_polygon(candidate_polygons)

            dist_scores = [
                float(np.mean(np.linalg.norm(poly - coarse, axis=1)))
                for poly in candidate_polygons
            ]
            cutoff = float(np.percentile(dist_scores, 75))
            filtered = [poly for poly, d in zip(candidate_polygons, dist_scores) if d <= cutoff]
            if not filtered:
                filtered = candidate_polygons
            final_poly = _median_polygon(filtered)
            if best_frame_for_refine is not None:
                final_poly = _refine_quad_to_lines(best_frame_for_refine, final_poly)
                final_poly = _calibrate_quad_scale_to_lines(best_frame_for_refine, final_poly)
                final_poly = _refine_quad_edges_to_lines(best_frame_for_refine, final_poly)
            mean_conf = float(np.mean(confs)) if confs else 0.0
            final_global_detection = {
                "type": "polygon",
                "points": [(float(x), float(y)) for x, y in final_poly],
                "area_points": [(float(x), float(y)) for x, y in final_poly],
                "class": target_class_name or "court skeleton",
                "conf": mean_conf,
                "is_locked": True,
                "is_stabilized": True,
            }
            frames_with_court = len(candidate_polygons)
            total_court_detections = len(candidate_polygons)

    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {source}")
    writer = _create_video_writer(target, fps, width, height)

    while True:
        ok, frame = capture.read()
        if not ok:
            break
        total_frames += 1

        detections: list[dict] = []
        if final_global_detection is not None:
            detections = [final_global_detection]
        elif normalized_provider == "local":
            detections = detect_local_frame(frame)
        else:
            detections = _roboflow_infer_frame(
                frame_bgr=frame,
                model_id=roboflow_model_id,
                api_key=roboflow_api_key,
                confidence_threshold=confidence_threshold,
            )
            if target_class_norm:
                detections = [
                    d
                    for d in detections
                    if _class_matches(target_class_norm, str(d.get("class", "")))
                ]

        if enable_stabilization and final_global_detection is None:
            if detections and detections[0].get("type") == "polygon":
                best_det = max(detections, key=lambda d: float(d.get("conf", 0.0)))
                curr_points = np.array(best_det.get("points", []), dtype=np.float32)
                accepted = False

                if curr_points.shape[0] >= 4:
                    if curr_points.shape[0] >= 12:
                        dst = DEFAULT_COURT_TEMPLATE_12[: curr_points.shape[0]]
                        err = _homography_reprojection_error(curr_points, dst)
                        accepted = err is not None and err <= reprojection_error_threshold
                    else:
                        accepted = True

                if accepted:
                    if (
                        last_good_keypoints is not None
                        and last_good_keypoints.shape == curr_points.shape
                    ):
                        curr_points = _smooth_keypoints(
                            prev_points=last_good_keypoints,
                            curr_points=curr_points,
                            alpha=smoothing_alpha,
                        )
                    best_det["points"] = [(float(x), float(y)) for x, y in curr_points]
                    best_det["area_points"] = _court_area_polygon_from_keypoints(curr_points)
                    best_det["is_stabilized"] = True

                    if static_camera_lock:
                        poly_np = np.array(best_det["area_points"], dtype=np.float32)
                        if poly_np.shape == (4, 2):
                            good_polygons_for_lock.append(poly_np)
                            if (
                                locked_polygon is None
                                and len(good_polygons_for_lock) >= max(2, lock_after_n_good_frames)
                            ):
                                locked_polygon = _median_polygon(good_polygons_for_lock)
                        if locked_polygon is not None:
                            best_det["area_points"] = [
                                (float(x), float(y)) for x, y in locked_polygon
                            ]
                            best_det["is_locked"] = True

                    last_good_keypoints = curr_points
                    last_good_detection = best_det.copy()
                    missing_streak = 0
                    detections = [best_det]
                else:
                    detections = []

            if (
                not detections
                and last_good_detection is not None
                and missing_streak < hold_last_good_frames
            ):
                missing_streak += 1
                held = last_good_detection.copy()
                held["is_held"] = True
                if static_camera_lock and locked_polygon is not None:
                    held["area_points"] = [
                        (float(x), float(y)) for x, y in locked_polygon
                    ]
                    held["is_locked"] = True
                detections = [held]

        if detections and final_global_detection is None:
            frames_with_court += 1
            total_court_detections += len(detections)

        for det in detections:
            _draw_court_detection_overlay(frame, det)

        writer.write(frame)
        if progress_callback is not None and total_frame_count > 0:
            if final_global_detection is not None:
                # second pass progress occupies 50%-100%
                progress_callback(
                    0.5 + 0.5 * min(1.0, total_frames / total_frame_count),
                    total_frames,
                    total_frame_count,
                )
                continue
            progress_callback(
                min(1.0, total_frames / total_frame_count),
                total_frames,
                total_frame_count,
            )

    capture.release()
    writer.release()

    return CourtStats(
        total_frames=total_frames,
        frames_with_court=frames_with_court,
        total_court_detections=total_court_detections,
        provider=normalized_provider,
        output_path=str(target),
    )
