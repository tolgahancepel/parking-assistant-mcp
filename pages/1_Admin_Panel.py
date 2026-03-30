"""
Admin Panel — human-in-the-loop reservation approval interface.

How it works (LangGraph interrupt pattern):
1. User completes reservation → chatbot runs notify_admin_node, then PAUSES
   before await_admin_approval_node (interrupt_before=["await_admin_approval"]).
2. This page reads pending reservations from the shared JSON store.
3. Admin clicks Approve or Reject →
   a. graph.update_state() injects approval_status into the paused graph state.
   b. graph.invoke(None, config) resumes → await_admin_approval_node runs →
      generates user-facing message → output_guard → conversation continues.
4. User's chat UI picks up the new message on next render.
"""

import streamlit as st

from graph.builder import get_graph
from store import pending_reservations as reservation_store

st.set_page_config(page_title="Admin Panel", page_icon="🔐", layout="centered")
st.title("🔐 Slytherin Admin Panel")
st.caption("Review and approve or reject parking reservation requests.")

graph = get_graph()  # same singleton as app.py — shares the MemorySaver checkpointer


# ---------------------------------------------------------------------------
# Decision handler
# ---------------------------------------------------------------------------

def process_decision(thread_id: str, decision: str) -> None:
    """
    Inject the admin decision into the paused LangGraph state and resume.

    1. Update the paused graph state with the admin's decision.
    2. Resume — await_admin_approval_node generates a user-facing message.
    3. pending_store is updated inside await_admin_approval_node itself.
    """
    config = {"configurable": {"thread_id": thread_id}}

    snapshot = graph.get_state(config)
    if not snapshot or "await_admin_approval" not in (snapshot.next or []):
        # Graph no longer waiting — just update the store
        st.warning("This reservation is no longer at the approval step.")
        reservation_store.set_status(thread_id, decision)
        return

    # Inject decision then resume
    graph.update_state(config, {"approval_status": decision})
    graph.invoke(None, config)

    label = "approved ✅" if decision == "approved" else "rejected ❌"
    st.success(f"Reservation {label}. The user's conversation has been updated.")


# ---------------------------------------------------------------------------
# Load and display pending reservations
# ---------------------------------------------------------------------------

pending = reservation_store.get_pending_all()

if not pending:
    st.success("✅ No pending reservations.")
    if st.button("🔄 Refresh"):
        st.rerun()
    st.stop()

st.subheader(f"Pending reservations: {len(pending)}")

if st.button("🔄 Refresh"):
    st.rerun()

for entry in pending:
    thread_id: str = entry["thread_id"]
    res: dict = entry["reservation"]
    token: str = entry.get("approval_token", "—")
    submitted_at: str = entry.get("submitted_at", "—")

    full_name = f"{res.get('name', '')} {res.get('surname', '')}".strip()

    with st.expander(f"📋 {full_name} — {res.get('car_number', '—')}", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Name:** {full_name}")
            st.markdown(f"**License plate:** `{res.get('car_number', '—')}`")
        with col2:
            st.markdown(f"**From:** {res.get('start_date', '—')}")
            st.markdown(f"**Until:** {res.get('end_date', '—')}")

        st.caption(f"Submitted: {submitted_at} | Token: `{token[:8]}…`")

        col_approve, col_reject = st.columns(2)

        with col_approve:
            if st.button("✅ Approve", key=f"approve_{thread_id}"):
                process_decision(thread_id, "approved")
                st.rerun()

        with col_reject:
            if st.button("❌ Reject", key=f"reject_{thread_id}"):
                process_decision(thread_id, "rejected")
                st.rerun()
