"""
LangGraph node functions for the parking assistant.

Each node:
- Accepts ParkingState
- Returns a dict of ONLY the keys it modifies (LangGraph merges the rest)
- Has no side effects beyond API calls and state updates

This makes every node independently unit-testable.
"""

from uuid import uuid4

from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI

from agents import admin_agent
from config import settings
from graph.state import ParkingState
from guardrails.filter import check_input, check_output
from notifications.notifier import send_reservation_notification
from rag.prompts import (
    EXTRACTION_PROMPT,
    INTENT_PROMPT,
    RAG_PROMPT,
    RESERVATION_STEP_MESSAGES,
    RESERVATION_STEPS,
)
from rag.retriever import retrieve
from store import pending_reservations as reservation_store


def _llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.openai_model,
        openai_api_key=settings.openai_api_key,
        temperature=0.3,
    )


# ---------------------------------------------------------------------------
# Node: input_guard
# ---------------------------------------------------------------------------

def input_guard_node(state: ParkingState) -> dict:
    """
    Extract the latest human message and validate it.
    If unsafe, add a refusal to messages so the graph can end gracefully.

    During reservation collection, only injection checks run (no topic check),
    because responses like "Alice" or "ABC-1234" are valid reservation field values
    that would otherwise look off-topic to an LLM topic filter.
    """
    # Extract raw text from the last HumanMessage
    last_human = next(
        (m for m in reversed(state["messages"]) if m.type == "human"), None
    )
    query = last_human.content if last_human else ""

    # In reservation mode, skip the LLM topic-relevance check
    in_reservation = bool(
        state.get("reservation_step") and state.get("reservation_step") != "complete"
    )
    is_safe, reason = check_input(query, skip_topic_check=in_reservation)

    updates: dict = {
        "user_query": query,
        "input_safe": is_safe,
        "guardrail_reason": reason if not is_safe else None,
    }

    if not is_safe:
        refusal = (
            "I'm sorry, I can only help with parking-related questions. "
            "Please ask me about Slytherin's rates, hours, location, or reservations."
        )
        updates["messages"] = [AIMessage(content=refusal)]
        updates["answer"] = refusal

    return updates


# ---------------------------------------------------------------------------
# Node: classify_intent
# ---------------------------------------------------------------------------

def classify_intent_node(state: ParkingState) -> dict:
    """Use the LLM to classify the user's intent."""
    response = _llm().invoke(
        INTENT_PROMPT.format_messages(messages=state["messages"])
    )
    raw = response.content.strip().lower()
    intent = raw if raw in ("info", "reservation") else "other"
    return {"intent": intent}


# ---------------------------------------------------------------------------
# Node: retrieve
# ---------------------------------------------------------------------------

def retrieve_node(state: ParkingState) -> dict:
    """Query Pinecone and return the top-K documents."""
    results = retrieve(state["user_query"])
    docs = [
        {
            "content": doc.page_content,
            "metadata": doc.metadata,
            "score": float(score),
        }
        for doc, score in results
    ]
    return {"retrieved_docs": docs}


# ---------------------------------------------------------------------------
# Node: generate
# ---------------------------------------------------------------------------

def generate_node(state: ParkingState) -> dict:
    """Generate a response using the retrieved context and conversation history."""
    context = "\n\n".join(d["content"] for d in state.get("retrieved_docs", []))
    response = _llm().invoke(
        RAG_PROMPT.format_messages(
            context=context,
            messages=state["messages"],
        )
    )
    # Do NOT add to messages here — output_guard_node finalises messages
    # to avoid duplicates with the add_messages reducer.
    return {"answer": response.content}


# ---------------------------------------------------------------------------
# Node: manage_reservation
# ---------------------------------------------------------------------------

def manage_reservation_node(state: ParkingState) -> dict:
    """
    Multi-turn reservation data collection.

    State machine:
      None        → ask for 'name'        (first time user expresses reservation intent)
      'name'      → extract name          → ask for 'surname'
      'surname'   → extract surname       → ask for 'car_number'
      'car_number'→ extract car_number    → ask for 'start_date'
      'start_date'→ extract start_date    → ask for 'end_date'
      'end_date'  → extract end_date      → step = 'complete', show summary
      'complete'  → reservation already done, redirect to info flow
    """
    step = state.get("reservation_step")
    reservation = dict(state.get("reservation") or {})

    # ── If we just entered reservation mode, start collecting ──
    if step is None:
        next_step = RESERVATION_STEPS[0]
        message = RESERVATION_STEP_MESSAGES[next_step]
        return {
            "reservation_step": next_step,
            "reservation": reservation,
            "answer": message,
        }

    # ── If already complete, treat as info query ──
    if step == "complete":
        message = (
            "Your reservation has already been submitted for approval. "
            "Please wait for administrator approval, or you can cancel the request."
        )
        return {"answer": message}

    # ── Extract the current field from the user's last message ──
    extraction_response = _llm().invoke(
        EXTRACTION_PROMPT.format_messages(
            field=step,
            user_message=state["user_query"],
        )
    )
    extracted = extraction_response.content.strip()

    if extracted == "NOT_FOUND":
        message = f"I didn't catch that. {RESERVATION_STEP_MESSAGES[step]}"
        return {"answer": message}

    # Store the extracted value
    reservation[step] = extracted

    # Advance to next step
    current_index = RESERVATION_STEPS.index(step)
    is_last = current_index == len(RESERVATION_STEPS) - 1

    if is_last:
        # All fields collected → show summary
        # Do NOT add to messages here — output_guard_node finalises messages.
        summary = _build_reservation_summary(reservation)
        return {
            "reservation": reservation,
            "reservation_step": "complete",
            "answer": summary,
            # Stage 2 will set approval_status = "pending" here
        }
    else:
        next_step = RESERVATION_STEPS[current_index + 1]
        message = RESERVATION_STEP_MESSAGES[next_step]
        return {
            "reservation": reservation,
            "reservation_step": next_step,
            "answer": message,
        }


def _build_reservation_summary(reservation: dict) -> str:
    return (
        "Thank you! Here is your reservation summary:\n\n"
        f"- **Name:** {reservation.get('name', '—')} {reservation.get('surname', '—')}\n"
        f"- **License plate:** {reservation.get('car_number', '—')}\n"
        f"- **From:** {reservation.get('start_date', '—')}\n"
        f"- **Until:** {reservation.get('end_date', '—')}\n\n"
        "Your reservation request has been submitted and is **pending administrator approval**. "
        "You will be notified once it is confirmed."
    )


# ---------------------------------------------------------------------------
# Node: notify_admin  (Stage 2)
# ---------------------------------------------------------------------------

def notify_admin_node(state: ParkingState) -> dict:
    """
    Triggered when reservation collection is complete.

    1. Admin agent formats a professional notification email.
    2. Notification is sent (email or file fallback).
    3. Reservation is saved to the pending store.
    4. A "pending approval" message is added to the conversation directly
       (not via output_guard, because the graph will pause before the next node).
    """
    reservation = state.get("reservation", {})
    thread_id = state.get("session_id", str(uuid4()))
    approval_token = str(uuid4())

    # Admin agent formats the notification
    subject, body = admin_agent.format_notification(reservation, approval_token)

    # Send via SMTP or log to file
    send_reservation_notification(
        subject=subject,
        body=body,
        smtp_host=settings.smtp_host,
        smtp_port=settings.smtp_port,
        smtp_user=settings.smtp_user,
        smtp_password=settings.smtp_password,
        admin_email=settings.admin_email,
    )

    # Persist in the shared pending store (admin panel reads this)
    reservation_store.add_pending(thread_id, approval_token, reservation)

    pending_message = (
        "Your reservation request has been **submitted to the administrator**. "
        "You will be notified once a decision is made.\n\n"
        "Please wait while the administrator is reviewing your request."
    )

    return {
        "approval_status": "pending",
        "approval_token": approval_token,
        "answer": pending_message,
        # Add directly — graph pauses after this node so output_guard won't run yet
        "messages": [AIMessage(content=pending_message)],
    }


# ---------------------------------------------------------------------------
# Node: await_admin_approval  (Stage 2)
# ---------------------------------------------------------------------------

def await_admin_approval_node(state: ParkingState) -> dict:
    """
    Runs after the admin has made a decision.

    The graph is compiled with interrupt_before=["await_admin_approval"],
    so this node only executes after the admin panel calls:
        graph.update_state(config, {"approval_status": "approved" | "rejected"})
        graph.invoke(None, config)

    The LLM (admin agent) generates a personalised user-facing message.
    """
    status = state.get("approval_status", "rejected")
    reservation = state.get("reservation", {})

    # Admin agent crafts the final user-facing decision message
    answer = admin_agent.format_decision_message(reservation, status)

    # Update the pending store so the admin panel reflects the final state
    thread_id = state.get("session_id")
    if thread_id:
        reservation_store.set_status(thread_id, status)

    return {"answer": answer}


# ---------------------------------------------------------------------------
# Node: output_guard
# ---------------------------------------------------------------------------

def output_guard_node(state: ParkingState) -> dict:
    """
    Validate the generated answer and add it to messages.

    This is the single place that appends the AI reply to the message list,
    which prevents duplicate entries from the add_messages reducer.
    If the answer is unsafe, it is replaced with a safe fallback first.

    During reservation collection the LLM check is skipped because
    prompts like "What is your license plate?" are always safe but
    can be misclassified by a generic safety LLM.
    """
    answer = state.get("answer", "")

    # Skip the expensive LLM check when collecting reservation fields
    in_reservation = bool(
        state.get("reservation_step") and state.get("reservation_step") != "complete"
    )
    is_safe, reason = check_output(answer, skip_llm_check=in_reservation)

    if not is_safe:
        answer = (
            "I'm sorry, I'm unable to provide that information. "
            "Please contact Slytherin customer service at info@Slytherin.example.com."
        )
        return {
            "output_safe": False,
            "guardrail_reason": reason,
            "answer": answer,
            "messages": [AIMessage(content=answer)],
        }

    return {
        "output_safe": True,
        "messages": [AIMessage(content=answer)],
    }
