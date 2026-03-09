"""
Sentence-transformer encoder wrapper for regulatory text embeddings.

Model: BAAI/bge-base-en-v1.5
  - 768-dim, 512-token context
  - Asymmetric: queries use an instruction prefix for better retrieval relevance
  - Trained on broad English corpus; strong on factual/regulatory retrieval (BEIR benchmarks)
  - Drop-in replacement for PubMedBERT — same 768-dim, no Qdrant schema change needed
"""

from __future__ import annotations

from sentence_transformers import SentenceTransformer

# BGE instruction prefix for query-side encoding (improves retrieval alignment)
_BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class RegulationEncoder:
    def __init__(
        self,
        model_name: str = "BAAI/bge-base-en-v1.5",
        device: str = "cpu",
        batch_size: int = 64,
    ) -> None:
        self.model = SentenceTransformer(model_name, device=device)
        self.batch_size = batch_size
        self.embedding_dim = self.model.get_sentence_embedding_dimension()

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
        """Encode a query string with BGE instruction prefix for asymmetric retrieval."""
        vector = self.model.encode(
            _BGE_QUERY_PREFIX + query,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return vector.tolist()
