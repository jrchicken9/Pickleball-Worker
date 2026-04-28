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
