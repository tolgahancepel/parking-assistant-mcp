"""
Tests for mcp_server/server.py

Uses FastAPI's TestClient — no real HTTP server needed.
Run with: pytest tests/test_mcp.py -v
"""

import os
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

# Set required env vars before importing the server module
os.environ.setdefault("MCP_API_KEY", "test-key")

from mcp_server.server import RESERVATIONS_FILE, app

client = TestClient(app)

VALID_HEADERS = {"X-API-Key": "test-key"}
SAMPLE_RESERVATION = {
    "name": "Alice Smith",
    "car_number": "ABC-1234",
    "start_date": "2025-07-01 09:00",
    "end_date": "2025-07-01 18:00",
}


# ---------------------------------------------------------------------------
# Auth tests
# ---------------------------------------------------------------------------

class TestAuthentication:
    def test_missing_api_key_is_rejected(self):
        response = client.post("/reservations", json=SAMPLE_RESERVATION)
        assert response.status_code in (401, 403)

    def test_wrong_api_key_returns_403(self):
        response = client.post(
            "/reservations",
            json=SAMPLE_RESERVATION,
            headers={"X-API-Key": "wrong-key"},
        )
        assert response.status_code == 403

    def test_valid_api_key_is_accepted(self, tmp_path):
        with patch("mcp_server.server.RESERVATIONS_FILE", tmp_path / "reservations.txt"):
            response = client.post(
                "/reservations",
                json=SAMPLE_RESERVATION,
                headers=VALID_HEADERS,
            )
        assert response.status_code == 201

    def test_health_endpoint_needs_no_auth(self):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"


# ---------------------------------------------------------------------------
# POST /reservations
# ---------------------------------------------------------------------------

class TestCreateReservation:
    def test_creates_entry_and_returns_201(self, tmp_path):
        file_path = tmp_path / "reservations.txt"
        with patch("mcp_server.server.RESERVATIONS_FILE", file_path):
            response = client.post(
                "/reservations",
                json=SAMPLE_RESERVATION,
                headers=VALID_HEADERS,
            )
        assert response.status_code == 201
        body = response.json()
        assert body["success"] is True
        assert "Alice Smith" in body["entry"]
        assert "ABC-1234" in body["entry"]
        assert "2025-07-01 09:00 → 2025-07-01 18:00" in body["entry"]

    def test_entry_written_to_file(self, tmp_path):
        file_path = tmp_path / "reservations.txt"
        with patch("mcp_server.server.RESERVATIONS_FILE", file_path):
            client.post("/reservations", json=SAMPLE_RESERVATION, headers=VALID_HEADERS)
        content = file_path.read_text()
        assert "Alice Smith" in content
        assert "ABC-1234" in content

    def test_entry_format_matches_spec(self, tmp_path):
        """Entry must follow: Name | Car Number | Period | Approval Time"""
        file_path = tmp_path / "reservations.txt"
        with patch("mcp_server.server.RESERVATIONS_FILE", file_path):
            client.post("/reservations", json=SAMPLE_RESERVATION, headers=VALID_HEADERS)
        line = file_path.read_text().strip()
        parts = [p.strip() for p in line.split("|")]
        assert len(parts) == 4
        assert parts[0] == "Alice Smith"
        assert parts[1] == "ABC-1234"
        assert "→" in parts[2]   # reservation period
        assert "UTC" in parts[3]  # approval time

    def test_multiple_reservations_appended(self, tmp_path):
        file_path = tmp_path / "reservations.txt"
        second = {**SAMPLE_RESERVATION, "name": "Bob Jones", "car_number": "XYZ-9999"}
        with patch("mcp_server.server.RESERVATIONS_FILE", file_path):
            client.post("/reservations", json=SAMPLE_RESERVATION, headers=VALID_HEADERS)
            client.post("/reservations", json=second, headers=VALID_HEADERS)
        lines = [l for l in file_path.read_text().splitlines() if l.strip()]
        assert len(lines) == 2


# ---------------------------------------------------------------------------
# GET /reservations
# ---------------------------------------------------------------------------

class TestListReservations:
    def test_returns_empty_when_no_file(self, tmp_path):
        with patch("mcp_server.server.RESERVATIONS_FILE", tmp_path / "missing.txt"):
            response = client.get("/reservations", headers=VALID_HEADERS)
        assert response.status_code == 200
        assert response.json()["total"] == 0

    def test_returns_all_entries(self, tmp_path):
        file_path = tmp_path / "reservations.txt"
        with patch("mcp_server.server.RESERVATIONS_FILE", file_path):
            client.post("/reservations", json=SAMPLE_RESERVATION, headers=VALID_HEADERS)
            response = client.get("/reservations", headers=VALID_HEADERS)
        assert response.json()["total"] == 1
        assert "Alice Smith" in response.json()["reservations"][0]
