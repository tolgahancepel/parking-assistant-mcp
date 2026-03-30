"""
Reservations table — shows all submitted reservations with their status.
"""

import pandas as pd
import streamlit as st

from store import pending_reservations as reservation_store

st.set_page_config(page_title="Reservations", page_icon="📋", layout="wide")
st.title("📋 Reservations")

if st.button("🔄 Refresh"):
    st.rerun()

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

rows = reservation_store.get_all()

if not rows:
    st.info("No reservations yet.")
    st.stop()

# ---------------------------------------------------------------------------
# Summary metrics
# ---------------------------------------------------------------------------

total = len(rows)
approved = sum(1 for r in rows if r["status"] == "approved")
pending = sum(1 for r in rows if r["status"] == "pending")
rejected = sum(1 for r in rows if r["status"] == "rejected")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total", total)
col2.metric("Approved", approved)
col3.metric("Pending", pending)
col4.metric("Rejected", rejected)

st.divider()

# ---------------------------------------------------------------------------
# Status filter
# ---------------------------------------------------------------------------

status_filter = st.selectbox(
    "Filter by status",
    options=["All", "pending", "approved", "rejected"],
    index=0,
)

filtered = rows if status_filter == "All" else [r for r in rows if r["status"] == status_filter]

# ---------------------------------------------------------------------------
# Table
# ---------------------------------------------------------------------------

STATUS_EMOJI = {"pending": "⏳ pending", "approved": "✅ approved", "rejected": "❌ rejected"}

df = pd.DataFrame(
    [
        {
            "Name": r["name"],
            "License plate": r["car_number"],
            "From": r["start_date"],
            "Until": r["end_date"],
            "Status": STATUS_EMOJI.get(r["status"], r["status"]),
            "Submitted at": r["submitted_at"].replace("T", " ").split(".")[0] + " UTC",
        }
        for r in filtered
    ]
)

st.dataframe(df, use_container_width=True, hide_index=True)
