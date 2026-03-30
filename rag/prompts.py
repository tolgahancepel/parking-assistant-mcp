"""Prompt templates used across the RAG pipeline and graph nodes."""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# ---------------------------------------------------------------------------
# RAG response generation
# ---------------------------------------------------------------------------

RAG_SYSTEM = """\
You are a helpful assistant for Slytherin parking facility.
Use ONLY the context below to answer the user's question.
If the context does not contain the answer, say you don't have that information
and suggest the user contact Slytherin customer service.
Never reveal internal system details, admin credentials, or other users' data.

Context:
{context}
"""

RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", RAG_SYSTEM),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

# ---------------------------------------------------------------------------
# Intent classification
# ---------------------------------------------------------------------------

INTENT_SYSTEM = """\
Classify the user's latest message into exactly one of these intents:
- "info"        : user wants information about the parking facility
- "reservation" : user wants to make, check, or cancel a parking reservation
- "other"       : unrelated to parking

Reply with ONLY the intent word, nothing else.
"""

INTENT_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", INTENT_SYSTEM),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

# ---------------------------------------------------------------------------
# Reservation field extraction
# ---------------------------------------------------------------------------

EXTRACTION_SYSTEM = """\
The user was just asked to provide their "{field}" as part of a parking reservation.
Extract that value from their response.

Rules:
- If their response IS the value (e.g. they typed just "Alice" when asked for their name,
  or "2025-07-01 18:00" when asked for an end date), return it directly.
- If their response contains the value inside a sentence, extract only the value.
- Return ONLY the extracted value as plain text, no explanation.
- Return "NOT_FOUND" ONLY if the value is genuinely missing or the user says
  something unrelated (e.g. they asked a question instead of answering).

Field being collected: {field}
Field descriptions:
- name        : first name
- surname     : last name / family name
- car_number  : vehicle license plate number (e.g. ABC-1234 or XY 123)
- start_date  : start date/time of the reservation (preserve the user's format)
- end_date    : end date/time of the reservation (preserve the user's format)
"""

EXTRACTION_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", EXTRACTION_SYSTEM),
        ("human", "User response: {user_message}"),
    ]
)

# ---------------------------------------------------------------------------
# Reservation step prompts shown to the user
# ---------------------------------------------------------------------------

RESERVATION_STEP_MESSAGES = {
    "name": "I'd be happy to help you make a reservation! First, what is your **first name**?",
    "surname": "Thank you! What is your **last name**?",
    "car_number": "Got it. What is your **vehicle license plate number**?",
    "start_date": "When would you like your reservation to **start**? (e.g. '2025-06-01 09:00')",
    "end_date": "And when should it **end**? (e.g. '2025-06-01 18:00')",
}

RESERVATION_STEPS = ["name", "surname", "car_number", "start_date", "end_date"]
