"""
Offline evaluation script for the RAG retriever.

Run with:
    python scripts/run_eval.py

Outputs a table of Precision@K, Recall@K, and MRR for each test query,
plus aggregate averages and per-query latency.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import settings
from evaluation.metrics import (
    EVAL_DATASET,
    measure_latency,
    mean_reciprocal_rank,
    precision_at_k,
    recall_at_k,
)
from rag.retriever import retrieve

K = settings.top_k


def run_evaluation() -> None:
    print(f"\n{'='*70}")
    print(f"RAG Evaluation  |  model={settings.embedding_model}  |  K={K}")
    print(f"{'='*70}\n")

    results = []

    for item in EVAL_DATASET:
        query = item["query"]
        relevant_ids = item["relevant_doc_ids"]

        (docs_with_scores, latency) = measure_latency(retrieve, query, K)

        retrieved_ids = [d.metadata.get("doc_id", "") for d, _ in docs_with_scores]

        p = precision_at_k(retrieved_ids, relevant_ids, K)
        r = recall_at_k(retrieved_ids, relevant_ids, K)
        mrr = mean_reciprocal_rank(retrieved_ids, relevant_ids)

        results.append({"query": query, "p@k": p, "r@k": r, "mrr": mrr, "latency": latency})

        print(f"Query   : {query}")
        print(f"Retrieved: {retrieved_ids}")
        print(f"Relevant : {relevant_ids}")
        print(f"P@{K}={p:.2f}  R@{K}={r:.2f}  MRR={mrr:.2f}  Latency={latency:.3f}s\n")

    avg_p = sum(r["p@k"] for r in results) / len(results)
    avg_r = sum(r["r@k"] for r in results) / len(results)
    avg_mrr = sum(r["mrr"] for r in results) / len(results)
    avg_lat = sum(r["latency"] for r in results) / len(results)

    print(f"{'='*70}")
    print(f"AVERAGES  P@{K}={avg_p:.2f}  R@{K}={avg_r:.2f}  MRR={avg_mrr:.2f}  Latency={avg_lat:.3f}s")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    run_evaluation()
