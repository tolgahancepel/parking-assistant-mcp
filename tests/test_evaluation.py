"""
Tests for evaluation/metrics.py

Pure unit tests — no external API calls.
Run with: pytest tests/test_evaluation.py -v
"""

import time

import pytest

from evaluation.metrics import (
    mean_reciprocal_rank,
    measure_latency,
    precision_at_k,
    recall_at_k,
)


class TestPrecisionAtK:
    def test_all_relevant(self):
        retrieved = ["doc1", "doc2", "doc3"]
        relevant = {"doc1", "doc2", "doc3"}
        assert precision_at_k(retrieved, relevant, k=3) == 1.0

    def test_none_relevant(self):
        retrieved = ["doc1", "doc2", "doc3"]
        relevant = {"doc4", "doc5"}
        assert precision_at_k(retrieved, relevant, k=3) == 0.0

    def test_half_relevant(self):
        retrieved = ["doc1", "doc2", "doc3", "doc4"]
        relevant = {"doc1", "doc3"}
        assert precision_at_k(retrieved, relevant, k=4) == 0.5

    def test_k_larger_than_retrieved(self):
        retrieved = ["doc1"]
        relevant = {"doc1"}
        # Only 1 doc retrieved but k=3: precision = 1/3
        assert precision_at_k(retrieved, relevant, k=3) == pytest.approx(1 / 3)

    def test_k_zero_returns_zero(self):
        assert precision_at_k(["doc1"], {"doc1"}, k=0) == 0.0

    def test_only_top_k_considered(self):
        # doc4 at position 4 should not count when k=3
        retrieved = ["doc1", "doc2", "doc3", "doc4"]
        relevant = {"doc4"}
        assert precision_at_k(retrieved, relevant, k=3) == 0.0


class TestRecallAtK:
    def test_full_recall(self):
        retrieved = ["doc1", "doc2"]
        relevant = {"doc1", "doc2"}
        assert recall_at_k(retrieved, relevant, k=2) == 1.0

    def test_zero_recall(self):
        retrieved = ["doc1", "doc2"]
        relevant = {"doc3"}
        assert recall_at_k(retrieved, relevant, k=2) == 0.0

    def test_partial_recall(self):
        retrieved = ["doc1", "doc2", "doc3"]
        relevant = {"doc1", "doc3", "doc5"}
        # 2 of 3 relevant found in top-3
        assert recall_at_k(retrieved, relevant, k=3) == pytest.approx(2 / 3)

    def test_empty_relevant_set(self):
        assert recall_at_k(["doc1"], set(), k=3) == 0.0

    def test_k_truncation(self):
        retrieved = ["doc1", "doc2", "doc3"]
        relevant = {"doc3"}
        # doc3 is at position 3; with k=2 it should not be counted
        assert recall_at_k(retrieved, relevant, k=2) == 0.0


class TestMeanReciprocalRank:
    def test_first_position(self):
        assert mean_reciprocal_rank(["doc1", "doc2"], {"doc1"}) == 1.0

    def test_second_position(self):
        assert mean_reciprocal_rank(["doc1", "doc2"], {"doc2"}) == pytest.approx(0.5)

    def test_third_position(self):
        assert mean_reciprocal_rank(["doc1", "doc2", "doc3"], {"doc3"}) == pytest.approx(1 / 3)

    def test_not_found(self):
        assert mean_reciprocal_rank(["doc1", "doc2"], {"doc99"}) == 0.0

    def test_multiple_relevant_uses_first(self):
        # First relevant doc is at rank 2; MRR = 0.5
        assert mean_reciprocal_rank(["doc1", "doc2", "doc3"], {"doc2", "doc3"}) == pytest.approx(0.5)


class TestMeasureLatency:
    def test_returns_result_and_float(self):
        result, latency = measure_latency(lambda: 42)
        assert result == 42
        assert isinstance(latency, float)
        assert latency >= 0.0

    def test_latency_is_positive(self):
        def slow_fn():
            time.sleep(0.05)
            return "done"

        _, latency = measure_latency(slow_fn)
        assert latency >= 0.05

    def test_args_and_kwargs_passed(self):
        def add(a, b=0):
            return a + b

        result, _ = measure_latency(add, 3, b=4)
        assert result == 7
