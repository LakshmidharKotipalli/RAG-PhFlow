"""
Retrieval quality evaluation against a golden set.

Requires running Qdrant + embedding model. Run with:
    pytest tests/eval/ -m integration -v

Reports Mean Precision@5 and Mean Recall@5 over the golden set.
Thresholds: P@5 >= 0.30, R@5 >= 0.40.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

GOLDEN_PATH = Path(__file__).parent / "golden_set.json"


@pytest.fixture(scope="module")
def golden_set() -> list[dict]:
    return json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def retriever():
    """Build a real HybridRetriever. Skips if infrastructure is unavailable."""
    try:
        from phflow.config.settings import settings
        from phflow.embeddings.encoder import RegulationEncoder
        from phflow.retrieval.hybrid import HybridRetriever
        from phflow.retrieval.reranker import Reranker
        from phflow.vectorstore.client import get_qdrant_client

        encoder = RegulationEncoder(
            model_name=settings.embedding_model,
            device=settings.embedding_device,
        )
        reranker = Reranker(
            model_name=settings.reranker_model,
            device=settings.embedding_device,
        )
        qdrant = get_qdrant_client()
        # Quick connectivity check
        qdrant.get_collections()

        return HybridRetriever(
            qdrant=qdrant,
            collection_name=settings.collection_name,
            encoder=encoder,
            reranker=reranker,
            initial_k=settings.initial_retrieval_k,
            reranker_top_n=settings.reranker_top_n,
            reranker_score_threshold=settings.reranker_score_threshold,
        )
    except Exception as exc:
        pytest.skip(f"Infrastructure unavailable: {exc}")


def _normalize(section_number: str) -> str:
    """Strip leading '§' and whitespace for comparison."""
    return section_number.lstrip("§ ").strip()


def _precision_at_k(retrieved: list[dict], expected: list[str], k: int = 5) -> float:
    top_k = [_normalize(s.get("section_number", "")) for s in retrieved[:k]]
    hits = sum(1 for exp in expected if any(exp in ret for ret in top_k))
    return hits / k if k else 0.0


def _recall_at_k(retrieved: list[dict], expected: list[str], k: int = 5) -> float:
    top_k = [_normalize(s.get("section_number", "")) for s in retrieved[:k]]
    hits = sum(1 for exp in expected if any(exp in ret for ret in top_k))
    return hits / len(expected) if expected else 0.0


@pytest.mark.integration
def test_precision_at_5(golden_set, retriever):
    """Mean Precision@5 across all golden queries must be >= 0.30."""
    scores: list[float] = []
    misses: list[str] = []

    for entry in golden_set:
        results = retriever.search(query=entry["query"])
        p = _precision_at_k(results, entry["expected_sections"])
        scores.append(p)
        if p == 0.0:
            misses.append(f"[{entry['id']}] {entry['query'][:60]}")

    mean_p5 = sum(scores) / len(scores) if scores else 0.0
    print(f"\nMean Precision@5: {mean_p5:.3f}  ({len(scores)} queries)")
    if misses:
        print(f"Zero-hit queries ({len(misses)}):")
        for m in misses:
            print(f"  {m}")

    assert mean_p5 >= 0.30, f"Mean Precision@5 {mean_p5:.3f} is below the 0.30 threshold"


@pytest.mark.integration
def test_recall_at_5(golden_set, retriever):
    """Mean Recall@5 across all golden queries must be >= 0.40."""
    scores: list[float] = []

    for entry in golden_set:
        results = retriever.search(query=entry["query"])
        r = _recall_at_k(results, entry["expected_sections"])
        scores.append(r)

    mean_r5 = sum(scores) / len(scores) if scores else 0.0
    print(f"\nMean Recall@5: {mean_r5:.3f}  ({len(scores)} queries)")

    assert mean_r5 >= 0.40, f"Mean Recall@5 {mean_r5:.3f} is below the 0.40 threshold"


@pytest.mark.integration
def test_per_query_report(golden_set, retriever, capsys):
    """Print a per-query breakdown for manual inspection (always passes)."""
    for entry in golden_set:
        results = retriever.search(query=entry["query"])
        p = _precision_at_k(results, entry["expected_sections"])
        r = _recall_at_k(results, entry["expected_sections"])
        retrieved_nums = [_normalize(s.get("section_number", "")) for s in results[:5]]
        status = "OK" if p > 0 else "MISS"
        print(
            f"[{status}] {entry['id']}  P@5={p:.2f}  R@5={r:.2f} | "
            f"expected={entry['expected_sections']} | got={retrieved_nums}"
        )
