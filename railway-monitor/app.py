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

refresh_sec = st.sidebar.slider("Auto-refresh (seconds)", min_value=2, max_value=30, value=5)
active_window_min = st.sidebar.slider("Active window (minutes)", min_value=1, max_value=60, value=10)
supabase_url = st.sidebar.text_input("SUPABASE_URL", value=os.getenv("SUPABASE_URL", ""))
supabase_key = st.sidebar.text_input("SUPABASE_KEY", value=os.getenv("SUPABASE_KEY", ""), type="password")
supabase_table = st.sidebar.text_input("SUPABASE_TABLE", value=os.getenv("SUPABASE_TABLE", "training_status"))
run_id = st.sidebar.text_input("RUN_ID (optional)", value=os.getenv("RUN_ID", ""))

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
            inactive: list[dict] = []
            for row in statuses:
                updated = float(row.get("updated_at_unix", 0.0))
                state = str(row.get("state", "unknown")).lower()
                if state == "running" or updated >= cutoff:
                    active.append(row)
                else:
                    inactive.append(row)

            st.subheader("Active Training Runs")
            if not active:
                st.info("No active runs in the selected time window.")
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

            with st.expander("Recent Inactive Runs", expanded=False):
                if not inactive:
                    st.caption("No inactive runs found.")
                else:
                    for row in inactive[:20]:
                        rid = str(row.get("run_id", "unknown"))
                        state = str(row.get("state", "unknown"))
                        epoch = int(row.get("epoch", 0))
                        epochs = max(1, int(row.get("epochs", 1)))
                        overall = float(row.get("overall_progress", 0.0))
                        st.write(f"- `{rid}` | {state} | epoch {epoch}/{epochs} | overall {overall*100:.1f}%")
    except Exception as exc:
        st.error(f"Supabase read failed: {exc}")

st.caption(f"Auto-refreshing in-app every {refresh_sec}s (no full page reload).")
