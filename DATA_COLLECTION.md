# Court Image Data Collection

This project includes a script to collect diverse pickleball-court images for model training.

## Install dependencies

```bash
python3 -m pip install -r requirements.txt
```

## Run collection

```bash
python3 scripts/collect_court_images.py
```

This saves images into:

- `data/raw/court_images/`

and metadata in:

- `data/raw/court_images/metadata.jsonl`

## Useful options

```bash
python3 scripts/collect_court_images.py \
  --max-per-query 150 \
  --min-width 800 \
  --min-height 450
```

Custom query examples:

```bash
python3 scripts/collect_court_images.py \
  --query "pickleball indoor court overhead" \
  --query "pickleball championship court camera"
```

## Notes

- The script deduplicates by SHA-256 hash.
- It filters invalid/small images automatically.
- Review and curate collected images before annotation/training.

## Collect frames from video URLs

1. Put URLs in `data/video_urls.txt` (one per line).
2. Run:

```bash
python3 scripts/extract_frames_from_video_urls.py
```

Useful options:

```bash
python3 scripts/extract_frames_from_video_urls.py \
  --fps 0.7 \
  --max-videos 20 \
  --min-width 960 \
  --min-height 540
```

If YouTube returns 403, run with browser cookies:

```bash
python3 scripts/extract_frames_from_video_urls.py \
  --cookies-from-browser safari
```

This script:

- downloads videos using `yt-dlp`
- samples frames at your requested FPS
- deduplicates against existing images by SHA-256
- appends frame metadata to `metadata.jsonl`

## Curate dataset (blur + duplicates)

After collecting images/frames, run:

```bash
python3 scripts/curate_images.py --copy-instead-of-move
```

Outputs:

- curated keep set: `data/curated/court_images/`
- rejected images: `data/curated/rejected/<reason>/`
- report: `data/curated/curation_report.jsonl`
- summary: `data/curated/curation_summary.json`

Useful options:

```bash
python3 scripts/curate_images.py \
  --blur-threshold 80 \
  --near-duplicate-hamming 5 \
  --min-width 800 \
  --min-height 450 \
  --copy-instead-of-move
```
