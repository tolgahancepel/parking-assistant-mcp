"""
Streamlit chat UI for the Slytherin parking assistant.

"""

import time
from uuid import uuid4

import streamlit as st
from langchain_core.messages import HumanMessage

from graph.builder import get_graph

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Slytherin Assistant", layout="centered")
st.title("🐍 Slytherin Parking Assistant")
st.caption("Ask me about parking rates, hours, location, or make a reservation.")

# ---------------------------------------------------------------------------
# Session initialisation
# ---------------------------------------------------------------------------

graph = get_graph()  # singleton — shared with admin panel

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid4())

if "first_invoke_done" not in st.session_state:
    st.session_state.first_invoke_done = False

thread_id: str = st.session_state.thread_id
config = {"configurable": {"thread_id": thread_id}}

# ---------------------------------------------------------------------------
# Load current graph state (checkpointer is source of truth)
# ---------------------------------------------------------------------------

snapshot = graph.get_state(config)
state_values: dict = snapshot.values if snapshot and snapshot.values else {}
messages: list = state_values.get("messages", [])
approval_status: str | None = state_values.get("approval_status")
is_awaiting_admin: bool = (
    bool(snapshot and "await_admin_approval" in (snapshot.next or []))
    or approval_status == "pending"
)

# ---------------------------------------------------------------------------
# Render conversation history
# ---------------------------------------------------------------------------

for msg in messages:
    role = "user" if msg.type == "human" else "assistant"
    with st.chat_message(role):
        st.markdown(msg.content)

# ---------------------------------------------------------------------------
# Admin-approval status banner
# ---------------------------------------------------------------------------

if is_awaiting_admin:
    st.info(
        "⏳ **Your reservation is awaiting administrator approval.** "
        "This page refreshes automatically every 3 seconds.",
        icon="🔔",
    )

# ---------------------------------------------------------------------------
# Sidebar: debug / session info
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Session Info")
    st.caption(f"`thread_id`: {thread_id[:8]}…")

    reservation = state_values.get("reservation", {})
    step = state_values.get("reservation_step")
    st.write(f"**Reservation step:** {step or 'None'}")

    if reservation:
        st.write("**Collected fields:**")
        for k, v in reservation.items():
            st.write(f"  - {k}: {v}")

    if approval_status:
        status_icon = {"pending": "⏳", "approved": "✅", "rejected": "❌"}.get(approval_status, "")
        st.write(f"**Approval status:** {status_icon} {approval_status}")

    docs = state_values.get("retrieved_docs", [])
    if docs:
        st.write(f"**Retrieved docs ({len(docs)}):**")
        for d in docs:
            st.write(f"  - {d['metadata'].get('doc_id', '?')} ({d['score']:.3f})")

    if st.button("Reset conversation"):
        st.session_state.thread_id = str(uuid4())
        st.session_state.first_invoke_done = False
        st.rerun()

# ---------------------------------------------------------------------------
# Chat input
# ---------------------------------------------------------------------------

if user_input := st.chat_input(
    "Type your message…",
    disabled=is_awaiting_admin,  # lock input while awaiting admin
):
    with st.chat_message("user"):
        st.markdown(user_input)

    # Build graph input
    if not st.session_state.first_invoke_done:
        # First invocation: provide all initial state fields
        graph_input = {
            "messages": [HumanMessage(content=user_input)],
            "session_id": thread_id,
            "user_query": "",
            "intent": None,
            "retrieved_docs": [],
            "answer": "",
            "input_safe": True,
            "output_safe": True,
            "guardrail_reason": None,
            "reservation": {},
            "reservation_step": None,
            "approval_status": None,
            "approval_token": None,
            "admin_response_payload": None,
            "reservation_confirmed": None,
            "reservation_file_path": None,
            "orchestration_trace": None,
            "active_subgraph": None,
        }
        st.session_state.first_invoke_done = True
    else:
        # Subsequent invocations: checkpointer handles merging saved state
        graph_input = {"messages": [HumanMessage(content=user_input)]}

    with st.spinner("Thinking…"):
        result = graph.invoke(graph_input, config=config)

    # Re-fetch snapshot to check if graph is now paused at interrupt
    snapshot = graph.get_state(config)
    now_awaiting = bool(snapshot and "await_admin_approval" in (snapshot.next or []))

    # Show the latest assistant reply
    answer = result.get("answer", "")
    if answer:
        with st.chat_message("assistant"):
            st.markdown(answer)

    # Guardrail warning
    if not result.get("input_safe", True) or not result.get("output_safe", True):
        reason = result.get("guardrail_reason", "")
        st.warning(f"⚠️ Guardrail triggered: {reason}", icon="🛡️")

    if now_awaiting:
        st.info(
            "⏳ **Reservation submitted!** Waiting for admin approval. "
            "Check the Admin Panel (sidebar) to see when a decision is made.",
            icon="🔔",
        )

    st.rerun()

# ---------------------------------------------------------------------------
# Auto-refresh while awaiting admin (after all UI is rendered)
# ---------------------------------------------------------------------------

if is_awaiting_admin:
    time.sleep(3)
    st.rerun()
