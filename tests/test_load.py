"""
Load tests for the parking assistant components.

Evaluates performance under concurrent / repeated load for:
  - MCP server          (concurrent reservation writes)
  - Chatbot RAG pipeline (concurrent graph sessions)
  - Admin approval path  (repeated approval resumptions)

All LLM / Pinecone calls are mocked so tests run without API keys.
Run with: pytest tests/test_load.py -v
"""

import concurrent.futures
import os
import time
from unittest.mock import MagicMock, patch

import pytest

os.environ.setdefault("MCP_API_KEY", "test-key")

from fastapi.testclient import TestClient

from mcp_server.server import app

client = TestClient(app)
HEADERS = {"X-API-Key": "test-key"}


# ---------------------------------------------------------------------------
# MCP server load tests
# ---------------------------------------------------------------------------

class TestMCPServerLoad:
    def test_concurrent_reservation_writes_all_succeed(self, tmp_path):
        """
        20 concurrent POST /reservations requests must all return 201.
        Verifies the server handles parallel writes without errors.
        """
        file_path = tmp_path / "concurrent_load.txt"
        N = 20

        def post(i):
            payload = {
                "name": f"LoadUser {i}",
                "car_number": f"LOAD-{i:04d}",
                "start_date": "2025-08-01 09:00",
                "end_date": "2025-08-01 18:00",
            }
            with patch("mcp_server.server.RESERVATIONS_FILE", file_path):
                return client.post("/reservations", json=payload, headers=HEADERS)

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(post, i) for i in range(N)]
            responses = [f.result() for f in concurrent.futures.as_completed(futures)]

        failed = [r.status_code for r in responses if r.status_code != 201]
        assert failed == [], f"{len(failed)} requests failed with statuses: {failed}"

    def test_mcp_server_average_latency(self, tmp_path):
        """
        Average response time for 10 sequential POST /reservations requests
        must stay under 500ms (pure in-process, no network).
        """
        file_path = tmp_path / "latency_load.txt"
        N = 10
        latencies = []

        for i in range(N):
            payload = {
                "name": f"LatUser {i}",
                "car_number": f"LAT-{i:04d}",
                "start_date": "2025-08-01 09:00",
                "end_date": "2025-08-01 18:00",
            }
            t0 = time.perf_counter()
            with patch("mcp_server.server.RESERVATIONS_FILE", file_path):
                response = client.post("/reservations", json=payload, headers=HEADERS)
            latencies.append(time.perf_counter() - t0)
            assert response.status_code == 201

        avg_ms = (sum(latencies) / N) * 1000
        assert avg_ms < 500, f"Average MCP latency {avg_ms:.1f}ms exceeded 500ms threshold"


# ---------------------------------------------------------------------------
# Chatbot RAG pipeline load tests
# ---------------------------------------------------------------------------

class TestChatbotLoad:
    @patch("graph.nodes.retrieve")
    @patch("graph.nodes._llm")
    @patch("guardrails.filter._llm")
    def test_concurrent_sessions_all_succeed(
        self, mock_guard_llm, mock_nodes_llm, mock_retrieve
    ):
        """
        10 concurrent chatbot sessions each sending a single info query
        must all complete successfully with input_safe=True.
        """
        from langchain_core.documents import Document
        from langchain_core.messages import HumanMessage
        from langgraph.checkpoint.memory import MemorySaver

        from graph.builder import build_graph

        guard_mock = MagicMock()
        guard_mock.invoke.return_value = MagicMock(content="SAFE")
        mock_guard_llm.return_value = guard_mock

        # Both classify and generate calls return sensible values
        nodes_mock = MagicMock()
        nodes_mock.invoke.return_value = MagicMock(content="info")
        mock_nodes_llm.return_value = nodes_mock

        mock_retrieve.return_value = [
            (Document(page_content="Mon–Fri 6am–11pm.", metadata={"doc_id": "hours_001"}), 0.9)
        ]

        g = build_graph().compile(
            checkpointer=MemorySaver(),
            interrupt_before=["await_admin_approval"],
        )
        N = 10

        def run(i):
            config = {"configurable": {"thread_id": f"load-session-{i}"}}
            return g.invoke(
                {"messages": [HumanMessage(content="What are the parking hours?")]},
                config,
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(run, i) for i in range(N)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        assert len(results) == N
        assert all(r["input_safe"] is True for r in results), \
            "Some sessions had input_safe=False unexpectedly"

    @patch("graph.nodes.retrieve")
    @patch("graph.nodes._llm")
    @patch("guardrails.filter._llm")
    def test_sequential_queries_within_time_budget(
        self, mock_guard_llm, mock_nodes_llm, mock_retrieve
    ):
        """
        5 sequential info queries (mocked) must complete within 10 seconds,
        demonstrating acceptable single-session throughput.
        """
        from langchain_core.documents import Document
        from langchain_core.messages import HumanMessage
        from langgraph.checkpoint.memory import MemorySaver

        from graph.builder import build_graph

        guard_mock = MagicMock()
        guard_mock.invoke.return_value = MagicMock(content="SAFE")
        mock_guard_llm.return_value = guard_mock

        nodes_mock = MagicMock()
        nodes_mock.invoke.return_value = MagicMock(content="info")
        mock_nodes_llm.return_value = nodes_mock

        mock_retrieve.return_value = [
            (Document(page_content="Rates: $3/hr.", metadata={"doc_id": "pricing_001"}), 0.85)
        ]

        g = build_graph().compile(
            checkpointer=MemorySaver(),
            interrupt_before=["await_admin_approval"],
        )

        t0 = time.perf_counter()
        for i in range(5):
            config = {"configurable": {"thread_id": f"throughput-{i}"}}
            g.invoke(
                {"messages": [HumanMessage(content="What is the location?")]},
                config,
            )
        elapsed = time.perf_counter() - t0

        assert elapsed < 10.0, f"5 sequential queries took {elapsed:.2f}s — exceeded 10s budget"


# ---------------------------------------------------------------------------
# Admin approval path load test
# ---------------------------------------------------------------------------

class TestAdminApprovalLoad:
    @patch("agents.admin_agent._llm")
    @patch("graph.nodes._llm")
    @patch("guardrails.filter._llm")
    def test_multiple_sequential_approvals(
        self, mock_guard_llm, mock_nodes_llm, mock_admin_llm
    ):
        """
        3 separate reservation threads approved one after another must all
        reach approval_status='approved' within a 30-second time budget.
        """
        from langchain_core.messages import HumanMessage
        from langgraph.checkpoint.memory import MemorySaver

        from graph.builder import build_graph

        guard_mock = MagicMock()
        guard_mock.invoke.return_value = MagicMock(content="SAFE")
        mock_guard_llm.return_value = guard_mock

        admin_mock = MagicMock()
        admin_mock.return_value = MagicMock(content="Your booking is confirmed.")
        mock_admin_llm.return_value = admin_mock

        checkpointer = MemorySaver()
        g = build_graph().compile(
            checkpointer=checkpointer,
            interrupt_before=["await_admin_approval"],
        )

        N = 3
        t0 = time.perf_counter()

        for idx in range(N):
            thread_id = f"approval-load-{idx}"
            config = {"configurable": {"thread_id": thread_id}}

            # Give each session its own side_effect list
            nodes_mock = MagicMock()
            nodes_mock.invoke.side_effect = [
                MagicMock(content="reservation"),
                MagicMock(content=f"User{idx}"),
                MagicMock(content="Load"),
                MagicMock(content=f"LOAD-{idx:04d}"),
                MagicMock(content="2025-09-01 09:00"),
                MagicMock(content="2025-09-01 17:00"),
                # After resume: output_guard doesn't call nodes._llm
            ]
            mock_nodes_llm.return_value = nodes_mock

            with patch("graph.nodes.send_reservation_notification"), \
                 patch("graph.nodes.reservation_store"):

                g.invoke({"messages": [HumanMessage(content="reserve")]}, config)
                for val in [
                    f"User{idx}", "Load", f"LOAD-{idx:04d}",
                    "2025-09-01 09:00", "2025-09-01 17:00",
                ]:
                    g.invoke({"messages": [HumanMessage(content=val)]}, config)

                # Admin approves
                g.update_state(config, {"approval_status": "approved"})

                # Resume node LLM calls: await_admin_approval uses admin_agent (already mocked)
                # output_guard LLM: step="complete" → in_reservation=False → calls guardrails LLM
                # guardrails mock already returns SAFE
                final = g.invoke(None, config)

            assert final["approval_status"] == "approved", \
                f"Thread {thread_id} ended with status={final['approval_status']}"

        elapsed = time.perf_counter() - t0
        assert elapsed < 30.0, f"{N} approval flows took {elapsed:.2f}s — exceeded 30s budget"