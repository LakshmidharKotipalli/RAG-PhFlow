"""
Sentence-transformer encoder wrapper for PubMedBERT embeddings.

Model: NeuML/pubmedbert-base-embeddings
  - 768-dim, 512-token context
  - Symmetric similarity (no instruction prefix needed)
  - Optimized for biomedical/pharmaceutical domain text
"""

from __future__ import annotations

from sentence_transformers import SentenceTransformer


class RegulationEncoder:
    def __init__(
        self,
        model_name: str = "NeuML/pubmedbert-base-embeddings",
        device: str = "cpu",
        batch_size: int = 64,
    ) -> None:
        self.model = SentenceTransformer(model_name, device=device)
        self.batch_size = batch_size
        self.embedding_dim = self.model.get_sentence_embedding_dimension()

    def encode_passages(self, texts: list[str], show_progress: bool = True) -> list[list[float]]:
        """Encode a list of passage texts into normalized embedding vectors."""
        vectors = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )
        return vectors.tolist()

    def encode_query(self, query: str) -> list[float]:
        """Encode a single query string into a normalized embedding vector."""
        vector = self.model.encode(
            query,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return vector.tolist()
