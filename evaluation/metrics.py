"""
RAG evaluation metrics.

All functions are pure: (data) -> float.
No LangGraph or Streamlit dependency — can be run offline via scripts/run_eval.py.

Metrics implemented:
- precision_at_k  : of top-K retrieved docs, what fraction are relevant
- recall_at_k     : of all relevant docs, what fraction appear in top-K
- mean_reciprocal_rank : average 1/rank of first relevant document
- measure_latency : wall-clock time for a retrieval+generation call
"""

import time
from typing import Callable


# ---------------------------------------------------------------------------
# Retrieval metrics
# ---------------------------------------------------------------------------

def precision_at_k(retrieved_doc_ids: list[str], relevant_doc_ids: set[str], k: int) -> float:
    """
    Precision@K = |relevant ∩ top-K retrieved| / K

    Args:
        retrieved_doc_ids: ordered list of doc IDs returned by the retriever
        relevant_doc_ids:  set of doc IDs considered ground-truth relevant
        k:                 number of top results to consider
    """
    if k <= 0:
        return 0.0
    top_k = retrieved_doc_ids[:k]
    hits = sum(1 for doc_id in top_k if doc_id in relevant_doc_ids)
    return hits / k


def recall_at_k(retrieved_doc_ids: list[str], relevant_doc_ids: set[str], k: int) -> float:
    """
    Recall@K = |relevant ∩ top-K retrieved| / |relevant|

    Args:
        retrieved_doc_ids: ordered list of doc IDs returned by the retriever
        relevant_doc_ids:  set of doc IDs considered ground-truth relevant
        k:                 number of top results to consider
    """
    if not relevant_doc_ids:
        return 0.0
    top_k = retrieved_doc_ids[:k]
    hits = sum(1 for doc_id in top_k if doc_id in relevant_doc_ids)
    return hits / len(relevant_doc_ids)


def mean_reciprocal_rank(retrieved_doc_ids: list[str], relevant_doc_ids: set[str]) -> float:
    """
    MRR = 1 / rank_of_first_relevant_document  (0 if none found)
    """
    for rank, doc_id in enumerate(retrieved_doc_ids, start=1):
        if doc_id in relevant_doc_ids:
            return 1.0 / rank
    return 0.0


# ---------------------------------------------------------------------------
# Latency measurement
# ---------------------------------------------------------------------------

def measure_latency(fn: Callable, *args, **kwargs) -> tuple:
    """
    Time a callable and return (result, elapsed_seconds).

    Usage:
        result, latency = measure_latency(retrieve, query, k=3)
    """
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed = time.perf_counter() - start
    return result, elapsed


# ---------------------------------------------------------------------------
# Evaluation dataset
# ---------------------------------------------------------------------------

# Ground-truth mapping: query → set of relevant doc_ids
# Used by scripts/run_eval.py to benchmark the retriever.
EVAL_DATASET = [
    {
        "query": "What are the parking rates?",
        "relevant_doc_ids": {"pricing_001"},
    },
    {
        "query": "Where is the parking located?",
        "relevant_doc_ids": {"location_001"},
    },
    {
        "query": "What time does the parking open?",
        "relevant_doc_ids": {"hours_001"},
    },
    {
        "query": "How do I make a reservation?",
        "relevant_doc_ids": {"booking_001"},
    },
    {
        "query": "Are there EV charging stations?",
        "relevant_doc_ids": {"spaces_001", "pricing_001"},
    },
    {
        "query": "What payment methods are accepted?",
        "relevant_doc_ids": {"payment_001"},
    },
    {
        "query": "Is there disabled parking?",
        "relevant_doc_ids": {"spaces_001"},
    },
    {
        "query": "What amenities are available?",
        "relevant_doc_ids": {"amenities_001"},
    },
]
