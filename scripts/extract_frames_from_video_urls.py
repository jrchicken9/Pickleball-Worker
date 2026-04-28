from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import requests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download videos from URLs and extract training frames."
    )
    parser.add_argument(
        "--urls-file",
        type=str,
        default="data/video_urls.txt",
        help="Text file with one video URL per line.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/raw/court_images",
        help="Directory where extracted frames are saved.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=0.5,
        help="Sampling rate in frames per second.",
    )
    parser.add_argument(
        "--min-width",
        type=int,
        default=640,
        help="Minimum frame width.",
    )
    parser.add_argument(
        "--min-height",
        type=int,
        default=360,
        help="Minimum frame height.",
    )
    parser.add_argument(
        "--max-videos",
        type=int,
        default=0,
        help="Optional cap on number of URLs to process (0 = all).",
    )
    parser.add_argument(
        "--cookies-from-browser",
        type=str,
        default="",
        help=(
            "Use yt-dlp browser cookies (e.g. safari, chrome, firefox, brave). "
            "Strongly recommended for YouTube."
        ),
    )
    return parser.parse_args()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_urls(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"URLs file not found: {path}")
    urls: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        urls.append(line)
    return urls


def load_seen_hashes(metadata_path: Path) -> set[str]:
    hashes: set[str] = set()
    if not metadata_path.exists():
        return hashes
    for line in metadata_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
            h = rec.get("sha256")
            if h:
                hashes.add(h)
        except json.JSONDecodeError:
            continue
    return hashes


def next_image_id(output_dir: Path) -> int:
    imgs = sorted(output_dir.glob("img_*.*"))
    return len(imgs) + 1


def run_yt_dlp(url: str, tmp_dir: Path, cookies_from_browser: str = "") -> Path:
    """
    Download best mp4/mov-friendly video for frame extraction.
    """
    output_tmpl = str(tmp_dir / "video.%(ext)s")
    is_youtube = "youtube.com" in url or "youtu.be" in url

    base_cmd = [
        "python3",
        "-m",
        "yt_dlp",
        "-f",
        "bv*+ba/best",
        "--merge-output-format",
        "mp4",
        "--no-playlist",
        "-o",
        output_tmpl,
    ]
    if is_youtube:
        base_cmd.extend(["--extractor-args", "youtube:player_client=tv,web"])

    attempts: list[list[str]] = []
    if cookies_from_browser:
        attempts.append(base_cmd + ["--cookies-from-browser", cookies_from_browser, url])
    attempts.append(base_cmd + [url])

    # Fallback browsers if explicit browser wasn't provided.
    if not cookies_from_browser:
        for browser in ["safari", "chrome", "firefox", "brave", "edge"]:
            attempts.append(base_cmd + ["--cookies-from-browser", browser, url])

    last_error: Exception | None = None
    for cmd in attempts:
        try:
            subprocess.run(cmd, check=True)
            break
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            continue
    else:
        raise RuntimeError(f"yt-dlp failed for URL {url}: {last_error}")

    candidates = sorted(tmp_dir.glob("video.*"))
    if not candidates:
        raise RuntimeError(f"yt-dlp did not produce a video file for URL: {url}")
    return candidates[0]


def download_direct_video(url: str, tmp_dir: Path) -> Path:
    """
    Download direct video file URLs (e.g. .mp4) without yt-dlp.
    """
    ext = ".mp4"
    lower = url.lower()
    for candidate in [".mp4", ".mov", ".mkv", ".avi", ".webm"]:
        if candidate in lower:
            ext = candidate
            break
    out_path = tmp_dir / f"video_direct{ext}"
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with out_path.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
    return out_path


def extract_frames(
    video_path: Path,
    fps: float,
    min_width: int,
    min_height: int,
    output_dir: Path,
    metadata_file,
    seen_hashes: set[str],
    start_id: int,
    source_url: str,
) -> int:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(video_fps / max(0.01, fps))))
    frame_idx = 0
    saved = 0
    next_id = start_id

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % step != 0:
            frame_idx += 1
            continue

        h, w = frame.shape[:2]
        if w < min_width or h < min_height:
            frame_idx += 1
            continue

        ok_enc, enc = cv2.imencode(".jpg", frame)
        if not ok_enc:
            frame_idx += 1
            continue
        img_bytes = enc.tobytes()
        digest = sha256_bytes(img_bytes)
        if digest in seen_hashes:
            frame_idx += 1
            continue

        file_name = f"img_{next_id:06d}.jpg"
        out_path = output_dir / file_name
        out_path.write_bytes(img_bytes)

        rec = {
            "id": next_id,
            "file": file_name,
            "source_type": "video_frame",
            "source_url": source_url,
            "video_file": video_path.name,
            "frame_index": frame_idx,
            "width": w,
            "height": h,
            "sha256": digest,
            "status": "accepted",
        }
        metadata_file.write(json.dumps(rec, ensure_ascii=True) + "\n")
        metadata_file.flush()

        seen_hashes.add(digest)
        next_id += 1
        saved += 1
        frame_idx += 1

    cap.release()
    return saved


def main() -> None:
    args = parse_args()
    urls_file = Path(args.urls_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "metadata.jsonl"

    urls = read_urls(urls_file)
    if args.max_videos > 0:
        urls = urls[: args.max_videos]

    seen_hashes = load_seen_hashes(metadata_path)
    next_id = next_image_id(output_dir)

    total_saved = 0
    with metadata_path.open("a", encoding="utf-8") as mf:
        for i, url in enumerate(urls, start=1):
            print(f"[{i}/{len(urls)}] Processing: {url}")
            with tempfile.TemporaryDirectory() as tmp:
                tmp_dir = Path(tmp)
                try:
                    if Path(url).exists():
                        video_path = Path(url)
                    elif url.lower().endswith((".mp4", ".mov", ".mkv", ".avi", ".webm")):
                        video_path = download_direct_video(url, tmp_dir)
                    else:
                        video_path = run_yt_dlp(
                            url, tmp_dir, cookies_from_browser=args.cookies_from_browser
                        )
                except Exception as exc:
                    print(f"  download failed: {exc}")
                    continue

                try:
                    saved = extract_frames(
                        video_path=video_path,
                        fps=args.fps,
                        min_width=args.min_width,
                        min_height=args.min_height,
                        output_dir=output_dir,
                        metadata_file=mf,
                        seen_hashes=seen_hashes,
                        start_id=next_id,
                        source_url=url,
                    )
                    total_saved += saved
                    next_id += saved
                    print(f"  saved frames: {saved}")
                except Exception as exc:
                    print(f"  frame extraction failed: {exc}")

                # Ensure temp files are gone.
                shutil.rmtree(tmp_dir, ignore_errors=True)

    print("\n=== Video frame extraction complete ===")
    print(f"saved_frames: {total_saved}")
    print(f"output: {output_dir}")
    print(f"metadata: {metadata_path}")


if __name__ == "__main__":
    main()
