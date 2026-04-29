from __future__ import annotations

import os
import time

import requests
import streamlit as st
from streamlit_autorefresh import st_autorefresh


def read_supabase_status_rows(
    *,
    supabase_url: str,
    supabase_key: str,
    table: str,
    run_id: str,
) -> list[dict]:
    endpoint = f"{supabase_url.rstrip('/')}/rest/v1/{table}"
    params = {
        "select": "*",
        "order": "updated_at_unix.desc",
        "limit": "100",
    }
    if run_id.strip():
        params["run_id"] = f"eq.{run_id.strip()}"
    headers = {
        "apikey": supabase_key,
        "Authorization": f"Bearer {supabase_key}",
    }
    resp = requests.get(endpoint, params=params, headers=headers, timeout=10)
    resp.raise_for_status()
    return resp.json()


st.set_page_config(page_title="Court Training Monitor", layout="centered")
st.title("Court Training Monitor")
st.caption("Railway monitor for local training status via Supabase (multi-run board).")

refresh_sec = int(os.getenv("MONITOR_REFRESH_SEC", "5"))
active_window_min = int(os.getenv("MONITOR_ACTIVE_WINDOW_MIN", "3"))
supabase_url = os.getenv("SUPABASE_URL", "")
supabase_key = os.getenv("SUPABASE_KEY", "")
supabase_table = os.getenv("SUPABASE_TABLE", "training_status")
run_id = os.getenv("RUN_ID", "")

st_autorefresh(interval=refresh_sec * 1000, key="monitor_refresh")

if not supabase_url.strip() or not supabase_key.strip():
    st.warning("Set SUPABASE_URL and SUPABASE_KEY environment variables in Railway.")
else:
    try:
        statuses = read_supabase_status_rows(
            supabase_url=supabase_url,
            supabase_key=supabase_key,
            table=supabase_table,
            run_id=run_id,
        )
        if not statuses:
            st.info("No training status rows found yet.")
        else:
            now = time.time()
            cutoff = now - (active_window_min * 60)

            active: list[dict] = []
            for row in statuses:
                updated = float(row.get("updated_at_unix", 0.0))
                # Consider a run "active" only if we've received an update within the last N minutes.
                # This prevents stale rows from lingering if the trainer stops uploading.
                if updated >= cutoff:
                    active.append(row)

            st.subheader("Active Training Runs")
            if not active:
                st.info("No active runs right now.")
            else:
                st.caption(f"Showing {len(active)} active runs.")
                for row in active:
                    rid = str(row.get("run_id", "unknown"))
                    state = str(row.get("state", "unknown"))
                    epoch = int(row.get("epoch", 0))
                    epochs = max(1, int(row.get("epochs", 1)))
                    elapsed = int(float(row.get("elapsed_seconds", 0.0)))
                    overall = float(row.get("overall_progress", 0.0))
                    epoch_prog = float(row.get("epoch_progress", 0.0))

                    st.markdown(f"**{rid}**  \nState: `{state}` | Epoch: `{epoch}/{epochs}` | Elapsed: `{elapsed // 60}m {elapsed % 60}s`")
                    st.progress(max(0.0, min(overall, 1.0)), text=f"Overall {overall*100:.1f}%")
                    st.progress(max(0.0, min(epoch_prog, 1.0)), text=f"Current epoch {epoch_prog*100:.1f}%")
                    st.divider()
    except Exception as exc:
        st.error(f"Supabase read failed: {exc}")

st.caption(f"Auto-refreshing in-app every {refresh_sec}s (no full page reload).")
