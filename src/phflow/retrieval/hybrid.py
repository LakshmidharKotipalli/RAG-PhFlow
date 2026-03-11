"""
Hybrid search: dense (FAISS) + BM25 sparse retrieval fused with
Reciprocal Rank Fusion (RRF), implemented client-side.

Flow:
  1. Encode query -> dense vector (BGE, with instruction prefix)
  2. FAISS inner-product search -> top candidates (dense)
  3. BM25 sparse scoring against stored sparse vectors -> top candidates (sparse)
  4. Client-side RRF fusion -> top-K merged candidates
  5. Parent expansion -> full section text for each top-N result
"""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np
from fastembed import SparseTextEmbedding

from phflow.settings import settings
from phflow.embeddings.encoder import RegulationEncoder
from phflow.vectorstore.client import FaissStore

logger = logging.getLogger(__name__)

# RRF constant (same as Qdrant default)
_RRF_K = 60


def _rrf_fuse(
    dense_results: list[dict],
    sparse_results: list[dict],
    k: int = _RRF_K,
    limit: int = 10,
) -> list[dict]:
    """
    Reciprocal Rank Fusion of dense and sparse result lists.

    Each result dict must have an 'id' key. The output carries merged
    score + the payload from whichever list had the entry (dense preferred).

    Returns top-`limit` results sorted by RRF score descending.
    """
    scores: dict[str, float] = {}
    payloads: dict[str, dict] = {}

    for rank, item in enumerate(dense_results):
        pid = item["id"]
        scores[pid] = scores.get(pid, 0.0) + 1.0 / (k + rank + 1)
        if pid not in payloads:
            payloads[pid] = item["payload"]

    for rank, item in enumerate(sparse_results):
        pid = item["id"]
        scores[pid] = scores.get(pid, 0.0) + 1.0 / (k + rank + 1)
        if pid not in payloads:
            payloads[pid] = item["payload"]

    # Sort by RRF score descending
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:limit]

    return [
        {"id": pid, "score": score, "payload": payloads[pid]}
        for pid, score in ranked
    ]


class HybridRetriever:
    def __init__(
        self,
        store: FaissStore,
        encoder: RegulationEncoder,
        initial_k: int | None = None,
        top_n: int | None = None,
    ) -> None:
        self.store = store
        self.encoder = encoder
        self.initial_k = initial_k or settings.initial_retrieval_k
        self.top_n = top_n or settings.retrieval_top_n
        self._sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")

    def _encode_sparse(self, text: str) -> dict:
        """Encode text to BM25 sparse vector as {indices, values}."""
        result = list(self._sparse_model.query_embed(text))[0]
        return {
            "indices": result.indices.tolist(),
            "values": result.values.tolist(),
        }

    def _sparse_search(
        self,
        query_sparse: dict,
        filter_fn: Callable[[dict], bool] | None = None,
        limit: int = 20,
    ) -> list[dict]:
        """
        Score all stored chunks against the query sparse vector using
        dot-product on overlapping token indices.

        Returns top-`limit` results sorted by score descending.
        """
        all_sparse = self.store.get_sparse_data()
        all_meta = self.store._metadata

        scored: list[tuple[float, int]] = []

        q_indices = set(query_sparse["indices"])
        q_map = dict(zip(query_sparse["indices"], query_sparse["values"]))

        for i, (sparse, meta) in enumerate(zip(all_sparse, all_meta)):
            if filter_fn and not filter_fn(meta):
                continue
            if not sparse or "indices" not in sparse:
                continue

            # Dot-product on overlapping indices
            score = 0.0
            for idx, val in zip(sparse["indices"], sparse["values"]):
                if idx in q_map:
                    score += val * q_map[idx]

            if score > 0:
                scored.append((score, i))

        # Sort by score descending, take top-limit
        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[:limit]

        return [
            {
                "id": all_meta[i].get("_point_id", str(i)),
                "score": s,
                "payload": all_meta[i],
            }
            for s, i in top
        ]

    def _build_authority_filter(
        self,
        user_filter: Callable[[dict], bool] | None,
    ) -> Callable[[dict], bool]:
        """Merge user filter with authority-exclusion filter."""
        def _combined(payload: dict) -> bool:
            if payload.get("chunk_type") == "authority":
                return False
            if user_filter and not user_filter(payload):
                return False
            return True
        return _combined

    def search(
        self,
        query: str,
        search_filter: Callable[[dict], bool] | None = None,
        expand_to_parent: bool = True,
        top_n: int | None = None,
        logger=None,
    ) -> list[dict]:
        """
        Retrieve relevant sections for a query using RRF-fused hybrid search.

        Returns a list of dicts, each representing one unique section with:
          - section_number, section_title, section_node
          - full_text (all chunks concatenated, for generation context)
          - score (RRF fusion score)
          - metadata (full payload from best matching chunk)
        """
        effective_top_n = top_n or self.top_n

        # Encode query
        dense_vec = self.encoder.encode_query(query)
        sparse_vec = self._encode_sparse(query)

        # Merge caller filter with the authority-exclusion filter
        combined_filter = self._build_authority_filter(search_filter)

        # Dense search via FAISS
        prefetch_limit = self.initial_k * settings.prefetch_multiplier
        dense_results = self.store.search(
            query_vector=dense_vec,
            limit=prefetch_limit,
            filter_fn=combined_filter,
        )

        # Sparse BM25 search
        sparse_results = self._sparse_search(
            query_sparse=sparse_vec,
            filter_fn=combined_filter,
            limit=prefetch_limit,
        )

        # RRF fusion
        candidates = _rrf_fuse(
            dense_results, sparse_results, limit=self.initial_k
        )

        if not candidates:
            if logger:
                logger.log("RETRIEVAL", "No candidates returned from FAISS.")
            return []

        # Take top-N by RRF score
        top_candidates = candidates[:effective_top_n]

        if logger:
            logger.log("RETRIEVAL (RRF top-N)", [
                {
                    "section_number": c["payload"].get("section_number", ""),
                    "section_title": c["payload"].get("section_title", ""),
                    "rrf_score": round(c["score"], 6),
                }
                for c in top_candidates
            ])

        if not expand_to_parent:
            return [
                {
                    "section_number": c["payload"].get("section_number", ""),
                    "section_title": c["payload"].get("section_title", ""),
                    "section_node": c["payload"].get("section_node", ""),
                    "full_text": c["payload"].get("chunk_text", ""),
                    "score": c["score"],
                    "metadata": c["payload"],
                }
                for c in top_candidates
            ]

        # Parent expansion: fetch full section for each unique section_node
        results: list[dict] = []
        seen_nodes: set[str] = set()

        for chunk in top_candidates:
            node = chunk["payload"].get("section_node", "")
            if node in seen_nodes:
                continue
            seen_nodes.add(node)

            siblings = self.store.scroll_by_node(node)

            # Concatenate sibling chunks into one full section text.
            # Keep the header from the first chunk only; strip it from
            # subsequent chunks so the same header does not repeat.
            texts: list[str] = []
            for i, s in enumerate(siblings):
                text = s["payload"].get("chunk_text", "")
                if i > 0:
                    lines = text.splitlines()
                    header_dropped = 0
                    start = 0
                    for j, line in enumerate(lines):
                        if header_dropped < 2:
                            if line.strip():
                                header_dropped += 1
                        else:
                            start = j
                            break
                    text = "\n".join(lines[start:]).lstrip("\n")
                texts.append(text)

            full_text = "\n\n".join(texts)

            results.append({
                "section_number": chunk["payload"].get("section_number", ""),
                "section_title": chunk["payload"].get("section_title", ""),
                "section_node": node,
                "full_text": full_text,
                "score": chunk["score"],
                "metadata": chunk["payload"],
            })

        return results
