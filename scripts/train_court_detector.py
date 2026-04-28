from __future__ import annotations

import argparse
import ast
import json
import time
import uuid
from pathlib import Path

import requests
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
        default="yolov8s.pt",
        help="Base YOLO model weights to fine-tune.",
    )
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs.")
    parser.add_argument(
        "--imgsz",
        type=int,
        default=960,
        help="Training image size (square).",
    )
    parser.add_argument("--batch", type=int, default=4, help="Batch size.")
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
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="DataLoader workers. If MPS shows instability, try 0.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=25,
        help="Early stopping patience in epochs.",
    )
    parser.add_argument(
        "--save-period",
        type=int,
        default=5,
        help="Save checkpoint every N epochs (-1 disables periodic saves).",
    )
    parser.add_argument(
        "--cache",
        action="store_true",
        help="Cache images in memory for faster training if RAM allows.",
    )
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable mixed precision (AMP).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume latest matching run instead of starting fresh.",
    )
    parser.add_argument(
        "--close-mosaic",
        type=int,
        default=10,
        help="Disable mosaic augmentation in last N epochs.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=50,
        help="Print batch progress every N batches.",
    )
    parser.add_argument(
        "--supabase-url",
        type=str,
        default="",
        help="Optional Supabase URL for remote status upload.",
    )
    parser.add_argument(
        "--supabase-key",
        type=str,
        default="",
        help="Optional Supabase publishable/anon key for status upload.",
    )
    parser.add_argument(
        "--supabase-table",
        type=str,
        default="training_status",
        help="Supabase table to upsert status into.",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default="",
        help="Run identifier used as Supabase row key.",
    )
    parser.add_argument(
        "--status-upload-interval",
        type=float,
        default=2.0,
        help="Min seconds between Supabase status uploads.",
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


def _fmt_time(seconds: float) -> str:
    s = max(0, int(seconds))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h > 0:
        return f"{h:d}:{m:02d}:{sec:02d}"
    return f"{m:02d}:{sec:02d}"


class StatusWriter:
    """Writes machine-readable local training status for monitors."""

    def __init__(self) -> None:
        self.path: Path | None = None
        self.started_at = time.time()

    def _ensure_path(self, trainer) -> Path:
        if self.path is not None:
            return self.path
        save_dir = Path(str(getattr(trainer, "save_dir", ".")))
        save_dir.mkdir(parents=True, exist_ok=True)
        self.path = save_dir / "train_status.json"
        return self.path

    def build_payload(
        self,
        trainer,
        *,
        state: str,
        epoch: int,
        epochs: int,
        batch: int,
        batches_per_epoch: int,
        epoch_progress: float,
        overall_progress: float,
    ) -> dict:
        now = time.time()
        return {
            "state": state,
            "epoch": int(epoch),
            "epochs": int(epochs),
            "batch": int(batch),
            "batches_per_epoch": int(batches_per_epoch),
            "epoch_progress": float(epoch_progress),
            "overall_progress": float(overall_progress),
            "save_dir": str(getattr(trainer, "save_dir", "")),
            "updated_at_unix": now,
            "elapsed_seconds": now - self.started_at,
        }

    def write(self, trainer, payload: dict) -> None:
        status_path = self._ensure_path(trainer)
        status_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


class SupabaseStatusUploader:
    """Uploads status snapshots to Supabase PostgREST."""

    def __init__(
        self,
        *,
        supabase_url: str,
        supabase_key: str,
        table: str,
        run_id: str,
        upload_interval_sec: float = 2.0,
    ) -> None:
        self.enabled = bool(supabase_url.strip() and supabase_key.strip())
        self.supabase_url = supabase_url.rstrip("/")
        self.supabase_key = supabase_key
        self.table = table
        self.run_id = run_id
        self.upload_interval_sec = max(0.2, upload_interval_sec)
        self.last_upload_ts = 0.0
        self._warned = False

    def upload(self, payload: dict, *, force: bool = False) -> None:
        if not self.enabled:
            return
        now = time.time()
        if not force and now - self.last_upload_ts < self.upload_interval_sec:
            return
        self.last_upload_ts = now

        endpoint = f"{self.supabase_url}/rest/v1/{self.table}?on_conflict=run_id"
        headers = {
            "apikey": self.supabase_key,
            "Authorization": f"Bearer {self.supabase_key}",
            "Content-Type": "application/json",
            "Prefer": "resolution=merge-duplicates,return=minimal",
        }
        body = {
            "run_id": self.run_id,
            "state": str(payload.get("state", "running")),
            "epoch": int(payload.get("epoch", 0)),
            "epochs": int(payload.get("epochs", 0)),
            "batch": int(payload.get("batch", 0)),
            "batches_per_epoch": int(payload.get("batches_per_epoch", 0)),
            "epoch_progress": float(payload.get("epoch_progress", 0.0)),
            "overall_progress": float(payload.get("overall_progress", 0.0)),
            "elapsed_seconds": float(payload.get("elapsed_seconds", 0.0)),
            "save_dir": str(payload.get("save_dir", "")),
            "updated_at_unix": float(payload.get("updated_at_unix", now)),
        }
        try:
            resp = requests.post(endpoint, headers=headers, json=body, timeout=10)
            if resp.status_code >= 300 and not self._warned:
                self._warned = True
                print(
                    f"[warn] Supabase upload failed ({resp.status_code}): {resp.text[:240]}",
                    flush=True,
                )
        except Exception as exc:
            if not self._warned:
                self._warned = True
                print(f"[warn] Supabase upload error: {exc}", flush=True)


class TrainingProgress:
    """Batch + epoch progress reporting with ETA."""

    def __init__(self, progress_every: int = 50) -> None:
        self.started = time.time()
        self.epoch_started = self.started
        self.progress_every = max(1, progress_every)
        self.last_batch_ts = self.started

    def on_epoch_start(self, trainer) -> None:
        self.epoch_started = time.time()
        self.last_batch_ts = self.epoch_started

    def on_batch_end(self, trainer) -> None:
        batch_i = int(getattr(trainer, "batch_i", -1))
        if batch_i < 0:
            return
        total_batches = len(getattr(trainer, "train_loader", []))
        if total_batches <= 0:
            return
        current = batch_i + 1
        if current % self.progress_every != 0 and current != total_batches:
            return

        now = time.time()
        epoch_elapsed = now - self.epoch_started
        sec_per_batch = epoch_elapsed / max(current, 1)
        epoch_eta = sec_per_batch * max(total_batches - current, 0)
        pct = 100.0 * current / total_batches
        width = 24
        filled = int(width * min(current / total_batches, 1.0))
        bar = "#" * filled + "-" * (width - filled)
        print(
            f"[batch] [{bar}] {current}/{total_batches} ({pct:5.1f}%) "
            f"it={sec_per_batch:.2f}s eta_epoch={_fmt_time(epoch_eta)}",
            flush=True,
        )
        self.last_batch_ts = now

    def on_epoch_end(self, trainer) -> None:
        done = int(getattr(trainer, "epoch", 0)) + 1
        total = int(getattr(trainer, "epochs", done))
        elapsed = time.time() - self.started
        avg = elapsed / max(done, 1)
        eta = avg * max(total - done, 0)
        width = 28
        filled = int(width * min(done / max(total, 1), 1.0))
        bar = "#" * filled + "-" * (width - filled)
        pct = 100.0 * done / max(total, 1)
        print(
            f"[train] [{bar}] {done}/{total} ({pct:5.1f}%) "
            f"elapsed={_fmt_time(elapsed)} eta={_fmt_time(eta)}",
            flush=True,
        )
        epoch_elapsed = time.time() - self.epoch_started
        print(
            f"[epoch] completed {done}/{total} in {_fmt_time(epoch_elapsed)}",
            flush=True,
        )


def main() -> None:
    args = parse_args()
    data_yaml = ensure_usable_data_yaml(resolve_data_yaml(args.data))
    model = YOLO(args.model)
    run_id = args.run_id.strip() or f"{args.name}-{uuid.uuid4().hex[:8]}"
    progress = TrainingProgress(progress_every=args.progress_every)
    status_writer = StatusWriter()
    uploader = SupabaseStatusUploader(
        supabase_url=args.supabase_url,
        supabase_key=args.supabase_key,
        table=args.supabase_table,
        run_id=run_id,
        upload_interval_sec=args.status_upload_interval,
    )

    def on_train_start(trainer) -> None:
        epochs = int(getattr(trainer, "epochs", 1))
        batches = len(getattr(trainer, "train_loader", []))
        payload = status_writer.build_payload(
            trainer,
            state="running",
            epoch=0,
            epochs=epochs,
            batch=0,
            batches_per_epoch=batches,
            epoch_progress=0.0,
            overall_progress=0.0,
        )
        status_writer.write(trainer, payload)
        uploader.upload(payload, force=True)

    def on_batch_end(trainer) -> None:
        epoch_idx = int(getattr(trainer, "epoch", 0))
        epochs = int(getattr(trainer, "epochs", 1))
        total_batches = max(1, len(getattr(trainer, "train_loader", [])))
        batch = int(getattr(trainer, "batch_i", -1)) + 1
        epoch_progress = min(max(batch / total_batches, 0.0), 1.0)
        overall = (epoch_idx + epoch_progress) / max(epochs, 1)
        payload = status_writer.build_payload(
            trainer,
            state="running",
            epoch=epoch_idx + 1,
            epochs=epochs,
            batch=batch,
            batches_per_epoch=total_batches,
            epoch_progress=epoch_progress,
            overall_progress=overall,
        )
        status_writer.write(trainer, payload)
        uploader.upload(payload, force=False)

    def on_epoch_end(trainer) -> None:
        epoch_idx = int(getattr(trainer, "epoch", 0))
        epochs = int(getattr(trainer, "epochs", 1))
        batches = len(getattr(trainer, "train_loader", []))
        payload = status_writer.build_payload(
            trainer,
            state="running",
            epoch=epoch_idx + 1,
            epochs=epochs,
            batch=0,
            batches_per_epoch=batches,
            epoch_progress=1.0,
            overall_progress=(epoch_idx + 1) / max(epochs, 1),
        )
        status_writer.write(trainer, payload)
        uploader.upload(payload, force=True)

    def on_train_end(trainer) -> None:
        epoch_idx = int(getattr(trainer, "epoch", 0))
        epochs = int(getattr(trainer, "epochs", 1))
        batches = len(getattr(trainer, "train_loader", []))
        payload = status_writer.build_payload(
            trainer,
            state="finished",
            epoch=min(epoch_idx + 1, epochs),
            epochs=epochs,
            batch=0,
            batches_per_epoch=batches,
            epoch_progress=1.0,
            overall_progress=1.0,
        )
        status_writer.write(trainer, payload)
        uploader.upload(payload, force=True)

    model.add_callback("on_train_start", on_train_start)
    model.add_callback("on_train_epoch_start", progress.on_epoch_start)
    model.add_callback("on_train_batch_end", on_batch_end)
    model.add_callback("on_train_batch_end", progress.on_batch_end)
    model.add_callback("on_train_epoch_end", on_epoch_end)
    model.add_callback("on_train_epoch_end", progress.on_epoch_end)
    model.add_callback("on_train_end", on_train_end)
    print(
        "Starting training with settings: "
        f"model={args.model}, imgsz={args.imgsz}, batch={args.batch}, "
        f"device={args.device}, epochs={args.epochs}, run_id={run_id}",
        flush=True,
    )
    results = model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        workers=args.workers,
        patience=args.patience,
        save_period=args.save_period,
        cache=args.cache,
        amp=not args.no_amp,
        resume=args.resume,
        close_mosaic=args.close_mosaic,
    )
    print(f"training output: {results.save_dir}")


if __name__ == "__main__":
    main()
