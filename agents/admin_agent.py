"""
Admin agent — the second LangChain agent in the system.

Responsibilities:
1. Format reservation approval requests for the human administrator
2. Generate user-facing messages for approved / rejected reservations

This agent is intentionally kept simple: it formats text using an LLM
and delegates the actual approve/reject decision to the human admin.
The human-in-the-loop pattern is enforced by the LangGraph interrupt
in the graph workflow (graph/builder.py).
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from config import settings


def _llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.openai_model,
        openai_api_key=settings.openai_api_key,
        temperature=0,
    )


# ---------------------------------------------------------------------------
# Prompt: format the admin notification email body
# ---------------------------------------------------------------------------

_NOTIFICATION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an assistant for Slytherin parking facility. "
            "Write a concise, professional reservation approval request email "
            "to be sent to the parking administrator. "
            "Include all reservation details clearly and end with clear "
            "instructions for the admin to approve or reject.",
        ),
        (
            "human",
            "Please write the email body for this reservation:\n\n"
            "Name: {name}\n"
            "License plate: {car_number}\n"
            "Period: {start_date} → {end_date}\n"
            "Approval token: {token}",
        ),
    ]
)

# ---------------------------------------------------------------------------
# Prompt: generate user-facing approval/rejection message
# ---------------------------------------------------------------------------

_DECISION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are the Slytherin parking assistant. "
            "Inform the customer of the administrator's decision in a friendly, "
            "professional tone. If approved, include next steps. "
            "If rejected, apologise and suggest contacting customer service.",
        ),
        (
            "human",
            "Reservation details:\n"
            "Name: {name}\n"
            "License plate: {car_number}\n"
            "Period: {start_date} → {end_date}\n\n"
            "Administrator decision: {decision}",
        ),
    ]
)


def format_notification(reservation: dict, approval_token: str) -> tuple[str, str]:
    """
    Format the admin notification email.

    Returns (subject, body).
    """
    name = reservation.get("name", "")
    surname = reservation.get("surname", "")
    full_name = f"{name} {surname}".strip()

    subject = f"[Slytherin] Reservation Approval Request — {full_name}"

    chain = _NOTIFICATION_PROMPT | _llm()
    body = chain.invoke(
        {
            "name": full_name,
            "car_number": reservation.get("car_number", "—"),
            "start_date": reservation.get("start_date", "—"),
            "end_date": reservation.get("end_date", "—"),
            "token": approval_token,
        }
    ).content

    return subject, body


def format_decision_message(reservation: dict, decision: str) -> str:
    """
    Generate a user-facing message for an approval or rejection.

    decision: "approved" | "rejected"
    """
    name = reservation.get("name", "")
    surname = reservation.get("surname", "")

    chain = _DECISION_PROMPT | _llm()
    return chain.invoke(
        {
            "name": f"{name} {surname}".strip(),
            "car_number": reservation.get("car_number", "—"),
            "start_date": reservation.get("start_date", "—"),
            "end_date": reservation.get("end_date", "—"),
            "decision": decision,
        }
    ).content
