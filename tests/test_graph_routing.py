"""
Tests for graph/builder.py routing functions.

Routing functions are pure state → edge-label functions with no external
calls, so no mocking is required.
Run with: pytest tests/test_graph_routing.py -v
"""

import pytest

from graph.builder import (
    route_after_classify,
    route_after_guard,
    route_after_reservation,
)


def make_state(**kwargs) -> dict:
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
    base.update(kwargs)
    return base


# ---------------------------------------------------------------------------
# route_after_guard
# ---------------------------------------------------------------------------

class TestRouteAfterGuard:
    def test_unsafe_input_routes_to_end(self):
        assert route_after_guard(make_state(input_safe=False)) == "end"

    def test_pending_approval_routes_to_check_status(self):
        assert route_after_guard(make_state(approval_status="pending")) == "check_approval_status"

    def test_mid_reservation_routes_to_manage(self):
        """Any active reservation step (not 'complete') goes to manage_reservation."""
        for step in ["name", "surname", "car_number", "start_date", "end_date"]:
            assert route_after_guard(make_state(reservation_step=step)) == "manage_reservation"

    def test_complete_step_routes_to_classify(self):
        """'complete' is treated as no active reservation — falls through to classify."""
        assert route_after_guard(make_state(reservation_step="complete")) == "classify_intent"

    def test_fresh_state_routes_to_classify(self):
        assert route_after_guard(make_state()) == "classify_intent"

    def test_unsafe_takes_priority_over_pending(self):
        """Unsafe flag must short-circuit before the approval_status check."""
        state = make_state(input_safe=False, approval_status="pending")
        assert route_after_guard(state) == "end"


# ---------------------------------------------------------------------------
# route_after_classify
# ---------------------------------------------------------------------------

class TestRouteAfterClassify:
    def test_reservation_intent_routes_to_manage(self):
        assert route_after_classify(make_state(intent="reservation")) == "manage_reservation"

    def test_info_intent_routes_to_retrieve(self):
        assert route_after_classify(make_state(intent="info")) == "retrieve"

    def test_other_intent_routes_to_retrieve(self):
        assert route_after_classify(make_state(intent="other")) == "retrieve"

    def test_none_intent_routes_to_retrieve(self):
        assert route_after_classify(make_state(intent=None)) == "retrieve"


# ---------------------------------------------------------------------------
# route_after_reservation
# ---------------------------------------------------------------------------

class TestRouteAfterReservation:
    def test_complete_step_routes_to_notify_admin(self):
        assert route_after_reservation(make_state(reservation_step="complete")) == "notify_admin"

    def test_incomplete_steps_route_to_output_guard(self):
        for step in ["name", "surname", "car_number", "start_date", "end_date"]:
            assert route_after_reservation(make_state(reservation_step=step)) == "output_guard"

    def test_none_step_routes_to_output_guard(self):
        assert route_after_reservation(make_state(reservation_step=None)) == "output_guard"