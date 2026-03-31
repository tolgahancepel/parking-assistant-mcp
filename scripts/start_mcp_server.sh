#!/bin/bash
# Start the MCP server. Run this before launching the Streamlit app.
# Usage: bash scripts/start_mcp_server.sh

set -a
source .env 2>/dev/null || true
set +a

uvicorn mcp_server.server:app --host 0.0.0.0 --port 8000
