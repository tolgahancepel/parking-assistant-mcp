"""
Tests for agents/admin_agent.py

Mocks all LLM calls so no API key is required.
Run with: pytest tests/test_admin_agent.py -v
"""

from unittest.mock import MagicMock, patch

import pytest

SAMPLE_RESERVATION = {
    "name": "Alice",
    "surname": "Smith",
    "car_number": "ABC-1234",
    "start_date": "2025-07-01 09:00",
    "end_date": "2025-07-01 18:00",
}


# ---------------------------------------------------------------------------
# format_notification tests
# ---------------------------------------------------------------------------

class TestFormatNotification:
    @patch("agents.admin_agent._llm")
    def test_returns_subject_and_body(self, mock_llm_fn):
        """format_notification must return a (subject, body) tuple of strings."""
        # LangChain LCEL wraps non-Runnable objects in RunnableLambda and calls them
        # as callables (mock_llm(messages)), so use mock_llm.return_value.content.
        mock_llm = MagicMock()
        mock_llm.return_value = MagicMock(content="Please review this reservation.")
        mock_llm_fn.return_value = mock_llm

        from agents.admin_agent import format_notification

        subject, body = format_notification(SAMPLE_RESERVATION, "token-123")

        assert isinstance(subject, str) and len(subject) > 0
        assert isinstance(body, str) and len(body) > 0

    @patch("agents.admin_agent._llm")
    def test_subject_includes_full_name(self, mock_llm_fn):
        """Subject line must contain the guest's full name."""
        mock_llm = MagicMock()
        mock_llm.return_value = MagicMock(content="Email body.")
        mock_llm_fn.return_value = mock_llm

        from agents.admin_agent import format_notification

        subject, _ = format_notification(SAMPLE_RESERVATION, "token-abc")

        assert "Alice Smith" in subject

    @patch("agents.admin_agent._llm")
    def test_body_is_llm_response(self, mock_llm_fn):
        """Body must be exactly the content returned by the LLM."""
        mock_llm = MagicMock()
        mock_llm.return_value = MagicMock(content="Custom email body text.")
        mock_llm_fn.return_value = mock_llm

        from agents.admin_agent import format_notification

        _, body = format_notification(SAMPLE_RESERVATION, "token-xyz")

        assert body == "Custom email body text."

    @patch("agents.admin_agent._llm")
    def test_missing_name_fields_do_not_raise(self, mock_llm_fn):
        """Reservation with missing name fields must not raise an exception."""
        mock_llm = MagicMock()
        mock_llm.return_value = MagicMock(content="Body.")
        mock_llm_fn.return_value = mock_llm

        from agents.admin_agent import format_notification

        subject, body = format_notification({}, "token-empty")

        assert isinstance(subject, str)
        assert isinstance(body, str)


# ---------------------------------------------------------------------------
# format_decision_message tests
# ---------------------------------------------------------------------------

class TestFormatDecisionMessage:
    @patch("agents.admin_agent._llm")
    def test_approved_returns_string(self, mock_llm_fn):
        """Approved decision must return a non-empty string."""
        mock_llm = MagicMock()
        mock_llm.return_value = MagicMock(content="Your reservation is approved!")
        mock_llm_fn.return_value = mock_llm

        from agents.admin_agent import format_decision_message

        result = format_decision_message(SAMPLE_RESERVATION, "approved")

        assert result == "Your reservation is approved!"

    @patch("agents.admin_agent._llm")
    def test_rejected_returns_string(self, mock_llm_fn):
        """Rejected decision must return a non-empty string."""
        mock_llm = MagicMock()
        mock_llm.return_value = MagicMock(
            content="Unfortunately your reservation was not approved."
        )
        mock_llm_fn.return_value = mock_llm

        from agents.admin_agent import format_decision_message

        result = format_decision_message(SAMPLE_RESERVATION, "rejected")

        assert isinstance(result, str) and len(result) > 0

    @patch("agents.admin_agent._llm")
    def test_llm_is_called_once(self, mock_llm_fn):
        """LLM must be called exactly once per format_decision_message call."""
        mock_llm = MagicMock()
        mock_llm.return_value = MagicMock(content="Decision message.")
        mock_llm_fn.return_value = mock_llm

        from agents.admin_agent import format_decision_message

        format_decision_message(SAMPLE_RESERVATION, "approved")

        # LCEL calls mock_llm(messages) (callable), not mock_llm.invoke(messages)
        mock_llm.assert_called_once()
