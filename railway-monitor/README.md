# Railway Monitor Deployment

Deploy this folder as the Railway service root directory.

## Why this folder exists

This monitor-only deployment avoids heavy ML dependencies (`opencv`, `ultralytics`, `torch`)
and avoids importing `src/worker.py`, so Railway builds are much faster and do not fail with
`libxcb.so.1` errors.

## Required Railway variables

- `SUPABASE_URL`
- `SUPABASE_KEY`
- `SUPABASE_TABLE` (optional, default: `training_status`)
- `RUN_ID` (optional)

## Root directory

Set Railway service **Root Directory** to:

`railway-monitor`
