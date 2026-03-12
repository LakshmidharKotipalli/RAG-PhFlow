"""
Semantic cache for RAG responses.

Embeds each query and compares against cached query embeddings using cosine
similarity.  When a new query is close enough to a cached one (above threshold),
the cached response is returned — skipping all LLM calls and retrieval.

Typical hit rate for a narrow domain like FDA dock-to-stock: 60-70%.
Latency on hit: ~5ms (one embedding + numpy dot product) vs ~3-5s full pipeline.
"""

from __future__ import annotations

import logging
import time

import numpy as np

logger = logging.getLogger(__name__)


class SemanticCache:
    """In-memory embedding-similarity cache with TTL eviction."""

    def __init__(
        self,
        threshold: float = 0.95,
        max_size: int = 512,
        ttl_seconds: int = 3600,
    ) -> None:
        self.threshold = threshold
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds

        # Parallel arrays — kept in sync by index.
        self._embeddings: list[np.ndarray] = []   # normalized query vectors
        self._responses: list[dict] = []           # cached pipeline responses
        self._timestamps: list[float] = []         # insertion time (epoch)
        self._queries: list[str] = []              # original query text (for logging)

    def _evict_expired(self) -> None:
        """Remove entries older than TTL."""
        now = time.time()
        cutoff = now - self.ttl_seconds
        keep = [i for i, ts in enumerate(self._timestamps) if ts > cutoff]
        if len(keep) == len(self._timestamps):
            return
        self._embeddings = [self._embeddings[i] for i in keep]
        self._responses = [self._responses[i] for i in keep]
        self._timestamps = [self._timestamps[i] for i in keep]
        self._queries = [self._queries[i] for i in keep]

    def lookup(self, query_embedding: list[float]) -> dict | None:
        """
        Check cache for a semantically similar query.

        Returns the cached response dict if similarity >= threshold, else None.
        """
        if not self._embeddings:
            return None

        self._evict_expired()
        if not self._embeddings:
            return None

        q_vec = np.array(query_embedding, dtype=np.float32)
        # Stack all cached embeddings into a matrix and compute cosine similarity
        # (vectors are already L2-normalized by the encoder).
        matrix = np.stack(self._embeddings)
        similarities = matrix @ q_vec

        best_idx = int(np.argmax(similarities))
        best_sim = float(similarities[best_idx])

        if best_sim >= self.threshold:
            logger.info(
                "Semantic cache HIT (sim=%.4f, query=%r -> cached=%r)",
                best_sim,
                "",  # caller can log the query separately
                self._queries[best_idx],
            )
            return self._responses[best_idx]

        return None

    def store(
        self,
        query: str,
        query_embedding: list[float],
        response: dict,
    ) -> None:
        """Add a query-response pair to the cache."""
        # Evict oldest if at capacity
        if len(self._embeddings) >= self.max_size:
            self._embeddings.pop(0)
            self._responses.pop(0)
            self._timestamps.pop(0)
            self._queries.pop(0)

        self._embeddings.append(np.array(query_embedding, dtype=np.float32))
        self._responses.append(response)
        self._timestamps.append(time.time())
        self._queries.append(query)

    @property
    def size(self) -> int:
        return len(self._embeddings)
