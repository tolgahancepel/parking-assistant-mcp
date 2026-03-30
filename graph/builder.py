"""
Assembles and compiles the LangGraph StateGraph.

Stage 2 graph (extends Stage 1):

  START
    │
    ▼
  input_guard ──(unsafe)──► END
    │
    │ (safe)
    ▼
  [router: approval_status?]
    ├── "pending"  ──► check_approval_status ──► output_guard ──► END
    └── other
          │
          ▼
        [router: in_reservation?]
          ├── yes ──► manage_reservation ──► [complete?]
          │                ├── yes ──► notify_admin ──► ◉ INTERRUPT ──► await_admin_approval ──► output_guard ──► END
          │                └── no  ──► output_guard ──► END
          └── no  ──► classify_intent
                          ├── "info"/"other" ──► retrieve ──► generate ──► output_guard ──► END
                          └── "reservation"  ──► manage_reservation ──► ...

◉ INTERRUPT = graph pauses here (interrupt_before=["await_admin_approval"])
  Resume: admin calls graph.update_state({approval_status: "approved"|"rejected"})
          then graph.invoke(None, config)

The MemorySaver checkpointer persists the full conversation state across
Streamlit reruns and between the user chat and admin panel.
"""

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from graph.nodes import (
    await_admin_approval_node,
    classify_intent_node,
    generate_node,
    input_guard_node,
    manage_reservation_node,
    notify_admin_node,
    output_guard_node,
    retrieve_node,
)
from graph.state import ParkingState

# Module-level singleton so both app.py and the admin panel share the same
# in-memory checkpointer — critical for the interrupt/resume pattern.
_checkpointer = MemorySaver()
_graph = None


# ---------------------------------------------------------------------------
# Routing functions
# ---------------------------------------------------------------------------

def route_after_guard(state: ParkingState) -> str:
    if not state.get("input_safe", True):
        return "end"
    # If awaiting admin decision, route to status check
    if state.get("approval_status") == "pending":
        return "check_approval_status"
    # If mid-reservation, continue collecting
    step = state.get("reservation_step")
    if step and step != "complete":
        return "manage_reservation"
    return "classify_intent"


def route_after_classify(state: ParkingState) -> str:
    return "manage_reservation" if state.get("intent") == "reservation" else "retrieve"


def route_after_reservation(state: ParkingState) -> str:
    """When reservation collection completes, escalate to admin."""
    if state.get("reservation_step") == "complete":
        return "notify_admin"
    return "output_guard"


# ---------------------------------------------------------------------------
# Inline node: check approval status
# ---------------------------------------------------------------------------

# TODO: This scenario may be deleted, never faced before
def check_approval_status_node(state: ParkingState) -> dict:
    """
    Called when user sends a message while approval_status == "pending".
    Re-checks the pending store in case the admin has already acted.
    """
    from langchain_core.messages import AIMessage
    from store import pending_reservations as reservation_store

    thread_id = state.get("session_id")
    current_status = reservation_store.get_status(thread_id) if thread_id else "pending"

    if current_status == "pending":
        msg = (
            "Your reservation is **still awaiting admin approval**. "
            "I'll let you know as soon as a decision is made. "
            "In the meantime, can I help you with anything else?"
        )
        return {"answer": msg}

    # Status was updated in the store (admin acted outside the graph interrupt)
    if current_status == "approved":
        msg = "Great news! Your reservation has been **approved** by the administrator!"
    else:
        msg = (
            "Unfortunately, your reservation was **not approved**. "
            "Please contact our customer service at info@Slytherin.example.com."
        )
    return {
        "approval_status": current_status,
        "answer": msg,
        "messages": [AIMessage(content=msg)],
    }


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    graph = StateGraph(ParkingState)

    # Stage 1 nodes
    graph.add_node("input_guard", input_guard_node)
    graph.add_node("classify_intent", classify_intent_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)
    graph.add_node("manage_reservation", manage_reservation_node)
    graph.add_node("output_guard", output_guard_node)

    # Stage 2 nodes
    graph.add_node("notify_admin", notify_admin_node)
    graph.add_node("await_admin_approval", await_admin_approval_node)
    graph.add_node("check_approval_status", check_approval_status_node)

    # Entry point
    graph.add_edge(START, "input_guard")

    # After guard: check for pending approval or route normally
    graph.add_conditional_edges(
        "input_guard",
        route_after_guard,
        {
            "end": END,
            "check_approval_status": "check_approval_status",
            "classify_intent": "classify_intent",
            "manage_reservation": "manage_reservation",
        },
    )

    # After intent classification
    graph.add_conditional_edges(
        "classify_intent",
        route_after_classify,
        {"retrieve": "retrieve", "manage_reservation": "manage_reservation"},
    )

    # RAG path
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "output_guard")

    # Reservation path: complete → notify admin; in-progress → output_guard
    graph.add_conditional_edges(
        "manage_reservation",
        route_after_reservation,
        {"notify_admin": "notify_admin", "output_guard": "output_guard"},
    )

    # Stage 2 path: notify → [INTERRUPT] → approval → output_guard
    graph.add_edge("notify_admin", "await_admin_approval")
    graph.add_edge("await_admin_approval", "output_guard")

    # Status check: if store already updated → add message directly
    graph.add_conditional_edges(
        "check_approval_status",
        lambda s: "end" if s.get("approval_status") in ("approved", "rejected") else "output_guard",
        {"end": END, "output_guard": "output_guard"},
    )

    # All paths end at output_guard → END
    graph.add_edge("output_guard", END)

    return graph


def get_graph():
    """Return the module-level singleton compiled graph (with checkpointer)."""
    global _graph
    if _graph is None:
        _graph = build_graph().compile(
            checkpointer=_checkpointer,
            interrupt_before=["await_admin_approval"],
        )
    return _graph


# Keep compile_graph() as an alias for backwards compatibility / tests
def compile_graph():
    return get_graph()
