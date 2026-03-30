"""
Tests for the RAG pipeline components.

Mocks all external calls (Pinecone, OpenAI) so tests run without API keys.
Run with: pytest tests/test_rag.py -v
"""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage

from graph.state import ParkingState


# ---------------------------------------------------------------------------
# Helper: minimal valid ParkingState
# ---------------------------------------------------------------------------

def make_state(**overrides) -> dict:
    base: dict = {
        "messages": [],
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
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Retrieve node tests
# ---------------------------------------------------------------------------

class TestRetrieveNode:
    @patch("graph.nodes.retrieve")
    def test_retrieve_node_populates_docs(self, mock_retrieve):
        """retrieve_node should populate retrieved_docs from Pinecone results."""
        from graph.nodes import retrieve_node

        mock_doc = Document(
            page_content="Parking rates are $3/hour.",
            metadata={"doc_id": "pricing_001", "category": "pricing"},
        )
        mock_retrieve.return_value = [(mock_doc, 0.92)]

        state = make_state(user_query="What are the parking rates?")
        result = retrieve_node(state)

        assert "retrieved_docs" in result
        assert len(result["retrieved_docs"]) == 1
        assert result["retrieved_docs"][0]["content"] == "Parking rates are $3/hour."
        assert result["retrieved_docs"][0]["score"] == pytest.approx(0.92)
        assert result["retrieved_docs"][0]["metadata"]["doc_id"] == "pricing_001"

    @patch("graph.nodes.retrieve")
    def test_retrieve_node_empty_results(self, mock_retrieve):
        """retrieve_node should handle empty retrieval gracefully."""
        from graph.nodes import retrieve_node

        mock_retrieve.return_value = []
        state = make_state(user_query="unknown query")
        result = retrieve_node(state)

        assert result["retrieved_docs"] == []


# ---------------------------------------------------------------------------
# Generate node tests
# ---------------------------------------------------------------------------

class TestGenerateNode:
    @patch("graph.nodes._llm")
    def test_generate_node_returns_answer(self, mock_llm_fn):
        """generate_node should call the LLM and return the answer text only.
        Messages are finalised by output_guard_node to avoid add_messages duplicates."""
        from graph.nodes import generate_node

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="Standard parking is $3/hour.")
        mock_llm_fn.return_value = mock_llm

        state = make_state(
            messages=[HumanMessage(content="What are the rates?")],
            retrieved_docs=[{"content": "Rates: $3/hour.", "metadata": {}, "score": 0.9}],
        )
        result = generate_node(state)

        assert result["answer"] == "Standard parking is $3/hour."
        assert "messages" not in result  # output_guard_node handles this

    @patch("graph.nodes._llm")
    def test_generate_node_uses_context(self, mock_llm_fn):
        """generate_node should include retrieved_docs in the prompt."""
        from graph.nodes import generate_node

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="Open 6am to 11pm.")
        mock_llm_fn.return_value = mock_llm

        state = make_state(
            messages=[HumanMessage(content="What are your hours?")],
            retrieved_docs=[{"content": "Hours: 6am-11pm Mon-Fri.", "metadata": {}, "score": 0.95}],
        )
        generate_node(state)

        # Verify the LLM was called (context is passed via prompt template)
        mock_llm.invoke.assert_called_once()


# ---------------------------------------------------------------------------
# Classify intent node tests
# ---------------------------------------------------------------------------

class TestClassifyIntentNode:
    @patch("graph.nodes._llm")
    def test_classify_info_intent(self, mock_llm_fn):
        from graph.nodes import classify_intent_node

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="info")
        mock_llm_fn.return_value = mock_llm

        state = make_state(messages=[HumanMessage(content="What are your hours?")])
        result = classify_intent_node(state)

        assert result["intent"] == "info"

    @patch("graph.nodes._llm")
    def test_classify_reservation_intent(self, mock_llm_fn):
        from graph.nodes import classify_intent_node

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="reservation")
        mock_llm_fn.return_value = mock_llm

        state = make_state(messages=[HumanMessage(content="I want to book a parking space.")])
        result = classify_intent_node(state)

        assert result["intent"] == "reservation"

    @patch("graph.nodes._llm")
    def test_classify_unknown_maps_to_other(self, mock_llm_fn):
        from graph.nodes import classify_intent_node

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="unknown_value")
        mock_llm_fn.return_value = mock_llm

        state = make_state(messages=[HumanMessage(content="blah blah")])
        result = classify_intent_node(state)

        assert result["intent"] == "other"


# ---------------------------------------------------------------------------
# Manage reservation node tests
# ---------------------------------------------------------------------------

class TestManageReservationNode:
    def test_first_call_asks_for_name(self):
        """When reservation_step is None, should start collecting name."""
        from graph.nodes import manage_reservation_node

        state = make_state(reservation_step=None, reservation={})
        result = manage_reservation_node(state)

        assert result["reservation_step"] == "name"
        assert "first name" in result["answer"].lower()

    @patch("graph.nodes._llm")
    def test_extracts_name_and_advances(self, mock_llm_fn):
        """When step='name' and user provides a name, should store it and move to surname.
        Messages are NOT added here — output_guard_node handles that."""
        from graph.nodes import manage_reservation_node

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="Alice")
        mock_llm_fn.return_value = mock_llm

        state = make_state(
            reservation_step="name",
            reservation={},
            user_query="My name is Alice",
        )
        result = manage_reservation_node(state)

        assert result["reservation"]["name"] == "Alice"
        assert result["reservation_step"] == "surname"
        assert "messages" not in result

    @patch("graph.nodes._llm")
    def test_not_found_asks_again(self, mock_llm_fn):
        """If extraction returns NOT_FOUND, should ask for the same field again."""
        from graph.nodes import manage_reservation_node

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="NOT_FOUND")
        mock_llm_fn.return_value = mock_llm

        state = make_state(
            reservation_step="name",
            reservation={},
            user_query="hello",
        )
        result = manage_reservation_node(state)

        assert result.get("reservation_step") == "name"  # stays on same step
        assert "didn't catch" in result["answer"].lower()

    @patch("graph.nodes._llm")
    def test_completion_shows_summary(self, mock_llm_fn):
        """When end_date is the last field, summary should be shown."""
        from graph.nodes import manage_reservation_node

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="2025-06-10 18:00")
        mock_llm_fn.return_value = mock_llm

        state = make_state(
            reservation_step="end_date",
            reservation={
                "name": "Alice",
                "surname": "Smith",
                "car_number": "ABC-123",
                "start_date": "2025-06-10 09:00",
            },
            user_query="2025-06-10 18:00",
        )
        result = manage_reservation_node(state)

        assert result["reservation_step"] == "complete"
        assert "summary" in result["answer"].lower()
        assert "pending" in result["answer"].lower()
