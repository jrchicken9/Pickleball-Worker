from __future__ import annotations

import os
from pathlib import Path

import requests
import streamlit as st

try:
    from streamlit_autorefresh import st_autorefresh
except ImportError:  # pragma: no cover
    st_autorefresh = None


RUNS_ROOT = Path("runs")


def _find_latest_run_dir(root: Path) -> Path | None:
    if not root.exists():
        return None
    candidates = [p.parent for p in root.rglob("args.yaml")]
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _read_latest_supabase_status(
    *,
    supabase_url: str,
    supabase_key: str,
    table: str,
    run_id: str,
) -> dict | None:
    endpoint = f"{supabase_url.rstrip('/')}/rest/v1/{table}"
    params = {
        "select": "*",
        "order": "updated_at_unix.desc",
        "limit": "1",
    }
    if run_id.strip():
        params["run_id"] = f"eq.{run_id.strip()}"
    headers = {
        "apikey": supabase_key,
        "Authorization": f"Bearer {supabase_key}",
    }
    resp = requests.get(endpoint, params=params, headers=headers, timeout=10)
    resp.raise_for_status()
    rows = resp.json()
    if not rows:
        return None
    return rows[0]


def _load_local_status(run_dir: Path) -> dict | None:
    status_path = run_dir / "train_status.json"
    if not status_path.exists():
        return None
    try:
        import json

        return json.loads(status_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _load_local_results_rows(run_dir: Path) -> list[dict]:
    results_path = run_dir / "results.csv"
    if not results_path.exists():
        return []
    lines = [ln.strip() for ln in results_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if len(lines) < 2:
        return []
    header = [h.strip() for h in lines[0].split(",")]
    rows: list[dict] = []
    for line in lines[1:]:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != len(header):
            continue
        row: dict[str, float | str] = {}
        for k, v in zip(header, parts):
            try:
                row[k] = float(v)
            except Exception:
                row[k] = v
        rows.append(row)
    return rows


st.set_page_config(page_title="Court Training Monitor", layout="centered")
st.title("Court Training Monitor")
st.caption("Railway web monitor for local YOLO training status via Supabase.")

refresh_sec = st.sidebar.slider("Auto-refresh (seconds)", min_value=2, max_value=30, value=5)
use_supabase_default = bool(os.getenv("SUPABASE_URL", "").strip() and os.getenv("SUPABASE_KEY", "").strip())
mode = st.sidebar.selectbox("Data source", ["Supabase", "Local files"] if use_supabase_default else ["Local files", "Supabase"])

if mode == "Supabase":
    supabase_url = st.sidebar.text_input("SUPABASE_URL", value=os.getenv("SUPABASE_URL", ""))
    supabase_key = st.sidebar.text_input("SUPABASE_KEY", value=os.getenv("SUPABASE_KEY", ""), type="password")
    supabase_table = st.sidebar.text_input("SUPABASE_TABLE", value=os.getenv("SUPABASE_TABLE", "training_status"))
    run_id = st.sidebar.text_input("RUN_ID (optional)", value=os.getenv("RUN_ID", ""))

    if not supabase_url.strip() or not supabase_key.strip():
        st.warning("Set SUPABASE_URL and SUPABASE_KEY to read remote status.")
    else:
        try:
            status = _read_latest_supabase_status(
                supabase_url=supabase_url,
                supabase_key=supabase_key,
                table=supabase_table,
                run_id=run_id,
            )
            if not status:
                st.info("No status rows found yet in Supabase.")
            else:
                epoch = int(status.get("epoch", 0))
                epochs = max(1, int(status.get("epochs", 1)))
                overall = float(status.get("overall_progress", 0.0))
                epoch_prog = float(status.get("epoch_progress", 0.0))
                state = str(status.get("state", "unknown"))
                elapsed = int(float(status.get("elapsed_seconds", 0.0)))
                run_val = str(status.get("run_id", ""))

                c1, c2, c3 = st.columns(3)
                c1.metric("State", state)
                c2.metric("Epoch", f"{epoch}/{epochs}")
                c3.metric("Elapsed", f"{elapsed // 60}m {elapsed % 60}s")
                st.write(f"Run ID: `{run_val}`")
                st.progress(max(0.0, min(overall, 1.0)), text=f"Overall progress: {overall*100:.1f}%")
                st.progress(max(0.0, min(epoch_prog, 1.0)), text=f"Epoch progress: {epoch_prog*100:.1f}%")
                st.json(status)
        except Exception as exc:
            st.error(f"Supabase read failed: {exc}")
else:
    target_run = st.sidebar.text_input("Optional run directory override", value="")
    if target_run.strip():
        run_dir = Path(target_run).expanduser()
    else:
        run_dir = _find_latest_run_dir(RUNS_ROOT)

    if run_dir is None or not run_dir.exists():
        st.warning("No training run found yet. Start training first.")
    else:
        st.write(f"Run: `{run_dir}`")
        status = _load_local_status(run_dir)
        rows = _load_local_results_rows(run_dir)
        if status:
            epoch = int(status.get("epoch", 0))
            epochs = max(1, int(status.get("epochs", 1)))
            overall = float(status.get("overall_progress", 0.0))
            epoch_prog = float(status.get("epoch_progress", 0.0))
            state = str(status.get("state", "unknown"))
            elapsed = int(float(status.get("elapsed_seconds", 0.0)))

            col1, col2, col3 = st.columns(3)
            col1.metric("State", state)
            col2.metric("Epoch", f"{epoch}/{epochs}")
            col3.metric("Elapsed", f"{elapsed // 60}m {elapsed % 60}s")
            st.progress(max(0.0, min(overall, 1.0)), text=f"Overall progress: {overall*100:.1f}%")
            st.progress(max(0.0, min(epoch_prog, 1.0)), text=f"Epoch progress: {epoch_prog*100:.1f}%")
        else:
            st.info("No local `train_status.json` found yet.")

        if rows:
            latest = rows[-1]
            m1, m2, m3 = st.columns(3)
            m1.metric("train/box_loss", f"{float(latest.get('train/box_loss', 0.0)):.4f}")
            m2.metric("train/cls_loss", f"{float(latest.get('train/cls_loss', 0.0)):.4f}")
            m3.metric("metrics/mAP50(B)", f"{float(latest.get('metrics/mAP50(B)', 0.0)):.4f}")

st.caption(f"Auto-refreshing every {refresh_sec}s")
if st_autorefresh is not None:
    st_autorefresh(interval=refresh_sec * 1000, key="training_monitor_refresh")
else:
    st.markdown(
        f"<meta http-equiv='refresh' content='{refresh_sec}'>",
        unsafe_allow_html=True,
    )
