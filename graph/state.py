"""
ParkingState — the single shared state for all stages.

Stage 1 fields are fully implemented.
Stage 2-4 fields are typed as Optional so they are inert
until the corresponding stages activate them.
"""

from typing import Annotated, Literal, Optional
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages


class ReservationData(TypedDict, total=False):
    """Fields collected during the reservation flow."""
    name: str
    surname: str
    car_number: str
    start_date: str
    end_date: str


class ParkingState(TypedDict):
    # ------------------------------------------------------------------ #
    # Stage 1: core RAG + reservation collection                          #
    # ------------------------------------------------------------------ #

    messages: Annotated[list, add_messages]
    # Full conversation history. add_messages appends; never overwrites.

    user_query: str
    # Raw string from the latest human turn (populated by input_guard).

    intent: Optional[Literal["info", "reservation", "other"]]
    # Classified intent of the current turn.

    retrieved_docs: list[dict]
    # List of {"content": str, "metadata": dict, "score": float} from Pinecone.

    answer: str
    # Final assistant response text (before it is added to messages).

    input_safe: bool
    # Set by input_guard. False → graph routes to a refusal.

    output_safe: bool
    # Set by output_guard. False → answer is replaced with a safe fallback.

    guardrail_reason: Optional[str]
    # Human-readable reason when a guardrail trips.

    reservation: ReservationData
    # Accumulated reservation fields collected across turns.

    reservation_step: Optional[str]
    # Current field being collected: "name" | "surname" | "car_number" |
    # "start_date" | "end_date" | "complete" | None (not in reservation mode).

    # ------------------------------------------------------------------ #
    # Stage 2: human-in-the-loop admin approval                          #
    # ------------------------------------------------------------------ #

    session_id: Optional[str]
    # Unique ID per user session (= LangGraph thread_id). Used by
    # notify_admin_node to key the pending reservation store.

    approval_status: Optional[Literal["pending", "approved", "rejected"]]
    # Set to "pending" by notify_admin_node; updated by admin panel.

    approval_token: Optional[str]
    # UUID generated at notification time; sent in the admin email.

    admin_response_payload: Optional[dict]

    # ------------------------------------------------------------------ #
    # Stage 3: MCP server file write (future)                             #
    # ------------------------------------------------------------------ #

    reservation_confirmed: Optional[bool]
    reservation_file_path: Optional[str]

    # ------------------------------------------------------------------ #
    # Stage 4: orchestration metadata (future)                            #
    # ------------------------------------------------------------------ #

    orchestration_trace: Optional[list[str]]
    active_subgraph: Optional[str]
