"""
Tests for store/pending_reservations.py

Uses a temporary file via monkeypatch so real store is never touched.
Run with: pytest tests/test_store.py -v
"""

import pytest

SAMPLE_RESERVATION = {
    "name": "Alice",
    "surname": "Smith",
    "car_number": "ABC-1234",
    "start_date": "2025-07-01 09:00",
    "end_date": "2025-07-01 18:00",
}


@pytest.fixture(autouse=True)
def isolated_store(tmp_path, monkeypatch):
    """Redirect all store I/O to a temporary file for each test."""
    import store.pending_reservations as store_module
    monkeypatch.setattr(store_module, "STORE_PATH", str(tmp_path / "test_store.json"))


# ---------------------------------------------------------------------------
# add_pending tests
# ---------------------------------------------------------------------------

class TestAddPending:
    def test_creates_pending_entry(self):
        """A freshly added reservation must have status 'pending'."""
        from store.pending_reservations import add_pending, get_status

        add_pending("thread-1", "token-abc", SAMPLE_RESERVATION)

        assert get_status("thread-1") == "pending"

    def test_stores_reservation_data(self):
        """Reservation fields must be persisted correctly."""
        from store.pending_reservations import _load, add_pending

        add_pending("thread-2", "token-xyz", SAMPLE_RESERVATION)

        store = _load()
        assert store["thread-2"]["reservation"]["name"] == "Alice"
        assert store["thread-2"]["approval_token"] == "token-xyz"

    def test_stores_submitted_at_timestamp(self):
        """Entry must include a submitted_at ISO timestamp."""
        from store.pending_reservations import _load, add_pending

        add_pending("thread-ts", "tok-ts", SAMPLE_RESERVATION)

        store = _load()
        submitted_at = store["thread-ts"]["submitted_at"]
        assert isinstance(submitted_at, str) and len(submitted_at) > 0


# ---------------------------------------------------------------------------
# set_status tests
# ---------------------------------------------------------------------------

class TestSetStatus:
    def test_updates_to_approved(self):
        from store.pending_reservations import add_pending, get_status, set_status

        add_pending("thread-3", "tok-1", SAMPLE_RESERVATION)
        set_status("thread-3", "approved")

        assert get_status("thread-3") == "approved"

    def test_updates_to_rejected(self):
        from store.pending_reservations import add_pending, get_status, set_status

        add_pending("thread-4", "tok-2", SAMPLE_RESERVATION)
        set_status("thread-4", "rejected")

        assert get_status("thread-4") == "rejected"

    def test_noop_for_unknown_thread(self):
        """set_status on a non-existent thread must not raise."""
        from store.pending_reservations import get_status, set_status

        set_status("nonexistent", "approved")

        assert get_status("nonexistent") is None


# ---------------------------------------------------------------------------
# get_pending_all tests
# ---------------------------------------------------------------------------

class TestGetPendingAll:
    def test_returns_only_pending_entries(self):
        from store.pending_reservations import add_pending, get_pending_all, set_status

        add_pending("t-pending", "tok-p", SAMPLE_RESERVATION)
        add_pending("t-approved", "tok-a", SAMPLE_RESERVATION)
        set_status("t-approved", "approved")

        pending = get_pending_all()
        thread_ids = [p["thread_id"] for p in pending]

        assert "t-pending" in thread_ids
        assert "t-approved" not in thread_ids

    def test_empty_when_no_pending(self):
        from store.pending_reservations import add_pending, get_pending_all, set_status

        add_pending("t-only", "tok", SAMPLE_RESERVATION)
        set_status("t-only", "approved")

        assert get_pending_all() == []


# ---------------------------------------------------------------------------
# get_all tests
# ---------------------------------------------------------------------------

class TestGetAll:
    def test_returns_all_statuses(self):
        from store.pending_reservations import add_pending, get_all, set_status

        add_pending("ta1", "tok1", SAMPLE_RESERVATION)
        add_pending("ta2", "tok2", SAMPLE_RESERVATION)
        set_status("ta2", "approved")

        rows = get_all()
        statuses = {r["thread_id"]: r["status"] for r in rows}

        assert statuses["ta1"] == "pending"
        assert statuses["ta2"] == "approved"

    def test_empty_store_returns_empty_list(self):
        from store.pending_reservations import get_all

        assert get_all() == []


# ---------------------------------------------------------------------------
# get_status tests
# ---------------------------------------------------------------------------

class TestGetStatus:
    def test_returns_none_for_missing_thread(self):
        from store.pending_reservations import get_status

        assert get_status("no-such-thread") is None

    def test_returns_pending_for_new_entry(self):
        from store.pending_reservations import add_pending, get_status

        add_pending("t-check", "tok-check", SAMPLE_RESERVATION)

        assert get_status("t-check") == "pending"