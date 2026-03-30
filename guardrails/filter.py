"""
Guardrail functions for input and output filtering.

Design:
- Pure functions: (text, ...) -> (is_safe: bool, reason: str)
- No LangGraph dependency; independently testable
- Two layers:
    1. Fast regex / keyword check (no LLM call)
    2. LLM semantic check for subtler cases
"""

import re

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from config import settings

# ---------------------------------------------------------------------------
# Sensitive patterns that must never appear in output
# ---------------------------------------------------------------------------

_SENSITIVE_PATTERNS = [
    r"admin[_\s]?pass(word)?",
    r"secret[_\s]?key",
    r"api[_\s]?key",
    r"database[_\s]?(url|password|host)",
    r"internal[_\s]?only",
    r"pinecone[_\s]?key",
    r"openai[_\s]?key",
]

# Prompt injection / jailbreak keywords (fast block, no LLM needed)
_INJECTION_KEYWORDS = [
    "hack", "exploit", "sql injection", "jailbreak", "ignore previous",
    "forget your instructions", "act as", "prompt injection",
]


def _llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.openai_model,
        openai_api_key=settings.openai_api_key,
        temperature=0,
    )


# ---------------------------------------------------------------------------
# Input guardrail
# ---------------------------------------------------------------------------

def check_input(text: str, skip_topic_check: bool = False) -> tuple[bool, str]:
    """
    Validate user input before it reaches the RAG pipeline.

    Args:
        text:             raw user message
        skip_topic_check: when True, only injection keywords are checked.
                          Set this during reservation collection where
                          short answers like "Alice" or "ABC-1234" are valid.

    Returns (is_safe, reason). reason is empty string when safe.
    """
    lower = text.lower()

    # 1. Fast keyword check (always runs)
    for keyword in _INJECTION_KEYWORDS:
        if keyword in lower:
            return False, f"Message contains disallowed content: '{keyword}'."

    if skip_topic_check:
        return True, ""

    # 2. LLM-based topic relevance (skipped during reservation collection)
    response = _llm().invoke(
        [
            SystemMessage(
                content=(
                    "You are a content moderator for a parking facility chatbot. "
                    "Decide if the user message is related to parking, reservations, "
                    "vehicles, or general customer service. "
                    "Reply with exactly 'SAFE' or 'UNSAFE: <one-line reason>'."
                )
            ),
            HumanMessage(content=text),
        ]
    )
    reply = response.content.strip()
    if reply.startswith("UNSAFE"):
        reason = reply.split(":", 1)[-1].strip()
        return False, reason

    return True, ""


# ---------------------------------------------------------------------------
# Output guardrail
# ---------------------------------------------------------------------------

def check_output(text: str, skip_llm_check: bool = False) -> tuple[bool, str]:
    """
    Validate LLM output before it is shown to the user.

    Args:
        text:           assistant response to validate
        skip_llm_check: when True, only regex patterns are checked.
                        Set this during reservation collection so that
                        prompts like "What is your license plate?" are never
                        incorrectly flagged as sensitive by the LLM.

    Returns (is_safe, reason). reason is empty string when safe.
    """
    lower = text.lower()

    # 1. Regex check for hardcoded sensitive patterns (always runs)
    for pattern in _SENSITIVE_PATTERNS:
        if re.search(pattern, lower):
            return False, f"Output contains sensitive pattern: '{pattern}'."

    if skip_llm_check:
        return True, ""

    # 2. LLM check for subtle leakage (skipped during reservation collection)
    response = _llm().invoke(
        [
            SystemMessage(
                content=(
                    "You are a security reviewer for a parking facility chatbot. "
                    "Check if the assistant response leaks any of the following:\n"
                    "- Admin credentials or API keys\n"
                    "- Internal system configuration details\n"
                    "- Personal data belonging to OTHER users (not the current user)\n\n"
                    "Note: asking the CURRENT user for their own name, car number, or "
                    "reservation dates is perfectly safe and should NOT be flagged.\n\n"
                    "Reply with exactly 'SAFE' or 'UNSAFE: <one-line reason>'."
                )
            ),
            HumanMessage(content=text),
        ]
    )
    reply = response.content.strip()
    if reply.startswith("UNSAFE"):
        reason = reply.split(":", 1)[-1].strip()
        return False, reason

    return True, ""
