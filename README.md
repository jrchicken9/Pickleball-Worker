# Human Detection Worker (Local Prototype)

This is a minimal local setup to test video human detection before adding more infrastructure.

It includes:
- A worker function that analyzes every frame in a video
- Person detection using a pre-trained YOLO model
- Pose estimation (body keypoints) to render skeletons into a new output video
- Optional highlighting of body joints that appear to be moving over time
- A tiny upload UI for quick testing

## 1) Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2) Run the local tester

```bash
streamlit run app.py
```

Then open the local URL printed by Streamlit, upload a video, and click **Run Detection**.

## Notes

- The first run downloads the model weights (`yolov8n.pt`), so it may be slower.
- Output video is encoded as MP4 with rendered tags/skeleton.
- In pose mode, the model highlights moving joints by comparing keypoints to the previous frame.
- If processing feels slow on CPU, try shorter clips while validating behavior.

## 3) Train a court detector from Roboflow export

If you export a YOLOv8 dataset ZIP from Roboflow, place it in this project and unzip it
under `data/datasets/roboflow`:

```bash
mkdir -p data/datasets/roboflow
unzip -o data/datasets/roboflow/pickleball_vision_yolov8.zip -d data/datasets/roboflow
```

Then train:

```bash
python3 scripts/train_court_detector.py \
  --data data/datasets/roboflow \
  --model yolov8m.pt \
  --epochs 100 \
  --imgsz 1280 \
  --batch 8 \
  --device mps
```

The best checkpoint is saved under `runs/pickleball_court/<run_name>/weights/best.pt`.

## 4) Railway monitor + local training (Supabase bridge)

Use local training for GPU/CPU work, and host only the monitor web app on Railway.

### Supabase table setup

Run this SQL in your Supabase SQL editor:

```sql
create table if not exists public.training_status (
  run_id text primary key,
  state text not null default 'running',
  epoch integer not null default 0,
  epochs integer not null default 0,
  batch integer not null default 0,
  batches_per_epoch integer not null default 0,
  epoch_progress double precision not null default 0,
  overall_progress double precision not null default 0,
  elapsed_seconds double precision not null default 0,
  save_dir text not null default '',
  updated_at_unix double precision not null default 0
);

alter table public.training_status enable row level security;

drop policy if exists "public read training_status" on public.training_status;
drop policy if exists "public insert training_status" on public.training_status;
drop policy if exists "public update training_status" on public.training_status;

create policy "public read training_status"
on public.training_status for select to anon using (true);

create policy "public insert training_status"
on public.training_status for insert to anon with check (true);

create policy "public update training_status"
on public.training_status for update to anon using (true) with check (true);
```

### Local training with status upload

```bash
python3 scripts/train_court_detector.py \
  --data data/datasets/roboflow/data.yaml \
  --supabase-url "https://YOUR_PROJECT.supabase.co" \
  --supabase-key "YOUR_PUBLISHABLE_OR_ANON_KEY"
```

### Monitor app

Local run:

```bash
streamlit run training_monitor_app.py
```

Railway run command:

```bash
streamlit run training_monitor_app.py --server.address 0.0.0.0 --server.port $PORT
```

Railway environment variables:
- `SUPABASE_URL`
- `SUPABASE_KEY`
- `SUPABASE_TABLE` (optional, default: `training_status`)
- `RUN_ID` (optional filter for a single active run)
