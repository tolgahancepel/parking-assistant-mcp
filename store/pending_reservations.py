"""
Simple JSON file-based store for pending reservations.

Provides a shared data contract between the user-facing chat (app.py)
and the admin panel (pages/1_Admin_Panel.py).

Schema per entry:
  {
    "<thread_id>": {
      "reservation": { name, surname, car_number, start_date, end_date },
      "approval_token": "<uuid>",
      "status": "pending" | "approved" | "rejected",
      "submitted_at": "<iso timestamp>"
    }
  }
"""

import json
import os
from datetime import datetime, timezone

STORE_PATH = os.path.join(os.path.dirname(__file__), "..", "pending_reservations.json")


def _load() -> dict:
    if not os.path.exists(STORE_PATH):
        return {}
    with open(STORE_PATH) as f:
        content = f.read().strip()
    if not content:
        return {}
    return json.loads(content)


def _save(data: dict) -> None:
    with open(STORE_PATH, "w") as f:
        json.dump(data, f, indent=2)


def add_pending(thread_id: str, approval_token: str, reservation: dict) -> None:
    """Record a new pending reservation."""
    store = _load()
    store[thread_id] = {
        "reservation": reservation,
        "approval_token": approval_token,
        "status": "pending",
        "submitted_at": datetime.now(timezone.utc).isoformat(),
    }
    _save(store)


def set_status(thread_id: str, status: str) -> None:
    """Update reservation status to 'approved' or 'rejected'."""
    store = _load()
    if thread_id in store:
        store[thread_id]["status"] = status
        _save(store)


def get_pending_all() -> list[dict]:
    """Return all reservations with status 'pending'."""
    store = _load()
    return [
        {"thread_id": tid, **entry}
        for tid, entry in store.items()
        if entry.get("status") == "pending"
    ]


def get_all() -> list[dict]:
    """Return all reservations regardless of status, newest first."""
    store = _load()
    rows = []
    for tid, entry in store.items():
        res = entry.get("reservation", {})
        rows.append({
            "name": f"{res.get('name', '')} {res.get('surname', '')}".strip(),
            "car_number": res.get("car_number", "—"),
            "start_date": res.get("start_date", "—"),
            "end_date": res.get("end_date", "—"),
            "status": entry.get("status", "—"),
            "submitted_at": entry.get("submitted_at", "—"),
            "thread_id": tid,
        })
    return sorted(rows, key=lambda r: r["submitted_at"], reverse=True)


def get_status(thread_id: str) -> str | None:
    """Return status for a given thread, or None if not found."""
    store = _load()
    return store.get(thread_id, {}).get("status")
