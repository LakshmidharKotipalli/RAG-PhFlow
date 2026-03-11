"""
Sentence-transformer encoder wrapper for regulatory text embeddings.

Model: BAAI/bge-base-en-v1.5
  - 768-dim, 512-token context
  - Asymmetric: queries use an instruction prefix for better retrieval relevance
  - Trained on broad English corpus; strong on factual/regulatory retrieval (BEIR benchmarks)

GPU acceleration:  The encoder attempts to use CUDA by default.  If no GPU
is available or the CUDA init fails, it falls back to CPU and logs a warning.
"""

from __future__ import annotations

import logging

from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# BGE instruction prefix for query-side encoding (improves retrieval alignment)
_BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# Simple LRU-style query vector cache — avoids re-encoding repeated queries.
# Keyed on the full prefixed string; evicts oldest entry when capacity is exceeded.
_QUERY_CACHE: dict[str, list[float]] = {}
_QUERY_CACHE_MAX = 256


def _resolve_device(requested: str) -> str:
    """
    Resolve the embedding device, preferring GPU with CPU fallback.

    If ``requested`` is ``"cuda"`` but no CUDA device is available,
    falls back to ``"cpu"`` and logs a warning.
    """
    if requested in ("cuda", "gpu"):
        try:
            import torch
            if torch.cuda.is_available():
                logger.info("Embedding device: CUDA GPU detected — using GPU")
                return "cuda"
            else:
                logger.warning(
                    "Embedding device: CUDA requested but no GPU available — "
                    "falling back to CPU"
                )
                return "cpu"
        except ImportError:
            logger.warning(
                "Embedding device: CUDA requested but torch not installed — "
                "falling back to CPU"
            )
            return "cpu"
    return requested


class RegulationEncoder:
    def __init__(
        self,
        model_name: str = "BAAI/bge-base-en-v1.5",
        device: str = "cuda",
        batch_size: int = 64,
    ) -> None:
        self._requested_device = device
        self.device = _resolve_device(device)
        self.model = SentenceTransformer(model_name, device=self.device)
        self.batch_size = batch_size
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        logger.info(
            "RegulationEncoder ready: model=%s, dim=%d, device=%s",
            model_name,
            self.embedding_dim,
            self.device,
        )

    def encode_passages(self, texts: list[str], show_progress: bool = True) -> list[list[float]]:
        """Encode passage texts (document side) — no prefix applied."""
        vectors = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )
        return vectors.tolist()

    def encode_query(self, query: str) -> list[float]:
        """Encode a query string with BGE instruction prefix for asymmetric retrieval.

        Results are cached by full prefixed string (up to _QUERY_CACHE_MAX entries).
        The model is deterministic, so repeated queries return the same vector.
        """
        cache_key = _BGE_QUERY_PREFIX + query
        if cache_key in _QUERY_CACHE:
            return _QUERY_CACHE[cache_key]

        vector = self.model.encode(
            cache_key,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        result = vector.tolist()

        if len(_QUERY_CACHE) >= _QUERY_CACHE_MAX:
            # Evict the oldest entry (insertion-order guaranteed in Python 3.7+)
            _QUERY_CACHE.pop(next(iter(_QUERY_CACHE)))
        _QUERY_CACHE[cache_key] = result
        return result
