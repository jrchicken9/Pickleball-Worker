from __future__ import annotations

import os

import requests
import streamlit as st


def read_latest_supabase_status(
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


st.set_page_config(page_title="Court Training Monitor", layout="centered")
st.title("Court Training Monitor")
st.caption("Railway monitor for local training status via Supabase.")

refresh_sec = st.sidebar.slider("Auto-refresh (seconds)", min_value=2, max_value=30, value=5)
supabase_url = st.sidebar.text_input("SUPABASE_URL", value=os.getenv("SUPABASE_URL", ""))
supabase_key = st.sidebar.text_input("SUPABASE_KEY", value=os.getenv("SUPABASE_KEY", ""), type="password")
supabase_table = st.sidebar.text_input("SUPABASE_TABLE", value=os.getenv("SUPABASE_TABLE", "training_status"))
run_id = st.sidebar.text_input("RUN_ID (optional)", value=os.getenv("RUN_ID", ""))

if not supabase_url.strip() or not supabase_key.strip():
    st.warning("Set SUPABASE_URL and SUPABASE_KEY environment variables in Railway.")
else:
    try:
        status = read_latest_supabase_status(
            supabase_url=supabase_url,
            supabase_key=supabase_key,
            table=supabase_table,
            run_id=run_id,
        )
        if not status:
            st.info("No training status rows found yet.")
        else:
            epoch = int(status.get("epoch", 0))
            epochs = max(1, int(status.get("epochs", 1)))
            overall = float(status.get("overall_progress", 0.0))
            epoch_prog = float(status.get("epoch_progress", 0.0))
            state = str(status.get("state", "unknown"))
            elapsed = int(float(status.get("elapsed_seconds", 0.0)))
            rid = str(status.get("run_id", ""))

            c1, c2, c3 = st.columns(3)
            c1.metric("State", state)
            c2.metric("Epoch", f"{epoch}/{epochs}")
            c3.metric("Elapsed", f"{elapsed // 60}m {elapsed % 60}s")
            st.write(f"Run ID: `{rid}`")
            st.progress(max(0.0, min(overall, 1.0)), text=f"Overall progress: {overall*100:.1f}%")
            st.progress(max(0.0, min(epoch_prog, 1.0)), text=f"Epoch progress: {epoch_prog*100:.1f}%")
            st.json(status)
    except Exception as exc:
        st.error(f"Supabase read failed: {exc}")

st.caption(f"Auto-refreshing every {refresh_sec}s")
st.markdown(
    f"<meta http-equiv='refresh' content='{refresh_sec}'>",
    unsafe_allow_html=True,
)
