from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


VALID_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


@dataclass
class ImageFeatures:
    path: Path
    sha256: str
    blur_score: float
    dhash: int
    width: int
    height: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Curate image dataset: remove blurry and near-duplicate images."
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="data/raw/court_images",
        help="Directory containing raw images.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/curated/court_images",
        help="Directory for curated images.",
    )
    parser.add_argument(
        "--reject-dir",
        type=str,
        default="data/curated/rejected",
        help="Directory for rejected images organized by reason.",
    )
    parser.add_argument(
        "--min-width",
        type=int,
        default=640,
        help="Minimum width to keep.",
    )
    parser.add_argument(
        "--min-height",
        type=int,
        default=360,
        help="Minimum height to keep.",
    )
    parser.add_argument(
        "--blur-threshold",
        type=float,
        default=70.0,
        help="Minimum variance of Laplacian to keep image.",
    )
    parser.add_argument(
        "--near-duplicate-hamming",
        type=int,
        default=6,
        help="Max dHash Hamming distance to consider near-duplicate.",
    )
    parser.add_argument(
        "--copy-instead-of-move",
        action="store_true",
        help="Copy files to curated/rejected dirs instead of moving them.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def variance_of_laplacian(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def dhash(gray: np.ndarray, hash_size: int = 8) -> int:
    resized = cv2.resize(gray, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    diff = resized[:, 1:] > resized[:, :-1]
    bits = 0
    bit_index = 0
    for row in diff:
        for bit in row:
            if bit:
                bits |= 1 << bit_index
            bit_index += 1
    return bits


def hamming_distance(a: int, b: int) -> int:
    return bin(int(a) ^ int(b)).count("1")


def read_image_features(path: Path) -> ImageFeatures | None:
    img = cv2.imread(str(path))
    if img is None:
        return None
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return ImageFeatures(
        path=path,
        sha256=sha256_file(path),
        blur_score=variance_of_laplacian(gray),
        dhash=dhash(gray),
        width=w,
        height=h,
    )


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def move_or_copy(src: Path, dst: Path, copy_only: bool) -> None:
    ensure_dir(dst.parent)
    if copy_only:
        shutil.copy2(src, dst)
    else:
        shutil.move(str(src), str(dst))


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    reject_dir = Path(args.reject_dir)
    ensure_dir(output_dir)
    ensure_dir(reject_dir)

    image_paths = sorted(
        p for p in input_dir.rglob("*") if p.is_file() and p.suffix.lower() in VALID_EXTS
    )
    if not image_paths:
        print(f"No images found in {input_dir}")
        return

    seen_sha: set[str] = set()
    kept: list[ImageFeatures] = []
    summary = {
        "input_count": len(image_paths),
        "kept_count": 0,
        "rejected_count": 0,
        "reasons": {
            "invalid_image": 0,
            "too_small": 0,
            "blurry": 0,
            "exact_duplicate": 0,
            "near_duplicate": 0,
        },
    }

    report_path = output_dir.parent / "curation_report.jsonl"
    with report_path.open("w", encoding="utf-8") as rf:
        for path in image_paths:
            rec = {"file": str(path)}
            feat = read_image_features(path)
            if feat is None:
                summary["rejected_count"] += 1
                summary["reasons"]["invalid_image"] += 1
                rec["status"] = "rejected"
                rec["reason"] = "invalid_image"
                dst = reject_dir / "invalid_image" / path.name
                move_or_copy(path, dst, args.copy_instead_of_move)
                rf.write(json.dumps(rec, ensure_ascii=True) + "\n")
                continue

            rec["width"] = feat.width
            rec["height"] = feat.height
            rec["blur_score"] = feat.blur_score
            rec["sha256"] = feat.sha256

            if feat.width < args.min_width or feat.height < args.min_height:
                summary["rejected_count"] += 1
                summary["reasons"]["too_small"] += 1
                rec["status"] = "rejected"
                rec["reason"] = "too_small"
                dst = reject_dir / "too_small" / path.name
                move_or_copy(path, dst, args.copy_instead_of_move)
                rf.write(json.dumps(rec, ensure_ascii=True) + "\n")
                continue

            if feat.blur_score < args.blur_threshold:
                summary["rejected_count"] += 1
                summary["reasons"]["blurry"] += 1
                rec["status"] = "rejected"
                rec["reason"] = "blurry"
                dst = reject_dir / "blurry" / path.name
                move_or_copy(path, dst, args.copy_instead_of_move)
                rf.write(json.dumps(rec, ensure_ascii=True) + "\n")
                continue

            if feat.sha256 in seen_sha:
                summary["rejected_count"] += 1
                summary["reasons"]["exact_duplicate"] += 1
                rec["status"] = "rejected"
                rec["reason"] = "exact_duplicate"
                dst = reject_dir / "exact_duplicate" / path.name
                move_or_copy(path, dst, args.copy_instead_of_move)
                rf.write(json.dumps(rec, ensure_ascii=True) + "\n")
                continue

            near_dup = False
            for k in kept:
                if hamming_distance(feat.dhash, k.dhash) <= args.near_duplicate_hamming:
                    near_dup = True
                    break

            if near_dup:
                summary["rejected_count"] += 1
                summary["reasons"]["near_duplicate"] += 1
                rec["status"] = "rejected"
                rec["reason"] = "near_duplicate"
                dst = reject_dir / "near_duplicate" / path.name
                move_or_copy(path, dst, args.copy_instead_of_move)
                rf.write(json.dumps(rec, ensure_ascii=True) + "\n")
                continue

            # Keep
            seen_sha.add(feat.sha256)
            kept.append(feat)
            summary["kept_count"] += 1
            rec["status"] = "kept"
            rec["reason"] = "ok"
            dst = output_dir / path.name
            move_or_copy(path, dst, args.copy_instead_of_move)
            rf.write(json.dumps(rec, ensure_ascii=True) + "\n")

    summary_path = output_dir.parent / "curation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"report: {report_path}")
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
