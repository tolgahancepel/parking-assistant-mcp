"""
MCP Server — Stage 3

A lightweight FastAPI server that processes confirmed parking reservations
and persists them to a text file.

Responsibilities:
- Expose POST /reservations to receive approved reservation data
- Authenticate all requests via X-API-Key header
- Append a formatted entry to confirmed_reservations.txt
- Provide GET /reservations to list all confirmed reservations

File entry format:
  Name | Car Number | Reservation Period | Approval Time

Run with:
    uvicorn mcp_server.server:app --host 0.0.0.0 --port 8000
"""

import os
from datetime import datetime, timezone
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Security, status
from fastapi.security import APIKeyHeader
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Configuration (reads from environment; defaults usable for local dev)
# ---------------------------------------------------------------------------

MCP_API_KEY = os.getenv("MCP_API_KEY", "dev-mcp-key")
RESERVATIONS_FILE = Path(os.getenv("RESERVATIONS_FILE", "confirmed_reservations.txt"))

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="Slytherin MCP Server", version="1.0.0")

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=True)


def verify_api_key(api_key: str = Security(_api_key_header)) -> str:
    if api_key != MCP_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key.",
        )
    return api_key


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class ReservationRequest(BaseModel):
    name: str
    car_number: str
    start_date: str
    end_date: str


class ReservationResponse(BaseModel):
    success: bool
    entry: str
    file_path: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_entry(req: ReservationRequest, approval_time: str) -> str:
    period = f"{req.start_date} → {req.end_date}"
    return f"{req.name} | {req.car_number} | {period} | {approval_time}"


def _append_to_file(entry: str) -> None:
    RESERVATIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with RESERVATIONS_FILE.open("a", encoding="utf-8") as f:
        f.write(entry + "\n")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post(
    "/reservations",
    response_model=ReservationResponse,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(verify_api_key)],
)
def create_reservation(req: ReservationRequest) -> ReservationResponse:
    """
    Write an approved reservation to the confirmed reservations file.
    Authenticated via X-API-Key header.
    """
    approval_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    entry = _format_entry(req, approval_time)
    _append_to_file(entry)
    return ReservationResponse(
        success=True,
        entry=entry,
        file_path=str(RESERVATIONS_FILE.resolve()),
    )


@app.get(
    "/reservations",
    dependencies=[Depends(verify_api_key)],
)
def list_reservations() -> dict:
    """Return all confirmed reservations from the file."""
    if not RESERVATIONS_FILE.exists():
        return {"reservations": [], "total": 0}
    lines = [
        line.strip()
        for line in RESERVATIONS_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return {"reservations": lines, "total": len(lines)}


@app.get("/health")
def health() -> dict:
    """Liveness probe — no auth required."""
    return {"status": "ok"}
