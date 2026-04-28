from __future__ import annotations

import argparse
import ast
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a YOLOv8 pickleball court detector from a dataset."
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/datasets/roboflow",
        help="Path to data.yaml or a directory containing data.yaml.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8m.pt",
        help="Base YOLO model weights to fine-tune.",
    )
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs.")
    parser.add_argument(
        "--imgsz",
        type=int,
        default=1280,
        help="Training image size (square).",
    )
    parser.add_argument("--batch", type=int, default=8, help="Batch size.")
    parser.add_argument(
        "--device",
        type=str,
        default="mps",
        help="Training device (e.g. cpu, mps, 0).",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="runs/pickleball_court",
        help="Output project directory for runs.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="roboflow_yolov8",
        help="Run name.",
    )
    return parser.parse_args()


def resolve_data_yaml(path_like: str) -> Path:
    candidate = Path(path_like)
    if candidate.is_file():
        if candidate.name != "data.yaml":
            raise ValueError(f"Expected data.yaml file, got: {candidate}")
        return candidate.resolve()

    if candidate.is_dir():
        direct = candidate / "data.yaml"
        if direct.exists():
            return direct.resolve()
        matches = sorted(candidate.rglob("data.yaml"))
        if len(matches) == 1:
            return matches[0].resolve()
        if not matches:
            raise ValueError(f"No data.yaml found under directory: {candidate}")
        raise ValueError(
            "Multiple data.yaml files found. "
            "Pass --data with the exact one to use."
        )

    raise ValueError(f"--data path does not exist: {candidate}")


def ensure_usable_data_yaml(data_yaml: Path) -> Path:
    """
    Make a training-ready YAML if an export only has train split.

    Some exports can contain only `train/images` while data.yaml still references
    `valid/images` and `test/images`. In that case, we generate a local fallback
    YAML that points val/test to train/images.
    """
    root = data_yaml.parent
    train_dir = root / "train" / "images"
    valid_dir = root / "valid" / "images"
    test_dir = root / "test" / "images"

    if valid_dir.exists():
        return data_yaml

    if not train_dir.exists():
        return data_yaml

    nc = 1
    names: list[str] = ["court"]
    for line in data_yaml.read_text(encoding="utf-8").splitlines():
        striped = line.strip()
        if striped.startswith("nc:"):
            try:
                nc = int(striped.split(":", 1)[1].strip())
            except Exception:
                pass
        elif striped.startswith("names:"):
            try:
                parsed = ast.literal_eval(striped.split(":", 1)[1].strip())
                if isinstance(parsed, list) and parsed:
                    names = [str(x) for x in parsed]
            except Exception:
                pass

    fallback_yaml = root / "data.autofix.yaml"
    fallback_yaml.write_text(
        (
            "train: ./train/images\n"
            "val: ./train/images\n"
            "test: ./train/images\n\n"
            f"nc: {nc}\n"
            f"names: {names}\n"
        ),
        encoding="utf-8",
    )
    print(
        "Generated fallback data config because valid/test split folders were missing: "
        f"{fallback_yaml}"
    )
    return fallback_yaml


def main() -> None:
    args = parse_args()
    data_yaml = ensure_usable_data_yaml(resolve_data_yaml(args.data))
    model = YOLO(args.model)
    results = model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
    )
    print(f"training output: {results.save_dir}")


if __name__ == "__main__":
    main()
