"""
Hybrid search: dense + BM25 sparse retrieval fused with Reciprocal Rank Fusion (RRF)
via Qdrant's native hybrid query API.

Flow:
  1. Encode query → dense vector (PubMedBERT) + sparse vector (BM25)
  2. Qdrant RRF fusion → top-K candidates
  3. Reranker → top-N sections
  4. Parent expansion → full section text for each top-N result
"""

from __future__ import annotations

from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Filter,
    Fusion,
    FusionQuery,
    Prefetch,
    SparseVector,
)

from phflow.embeddings.encoder import RegulationEncoder
from phflow.retrieval.reranker import Reranker
from phflow.vectorstore.client import scroll_by_node


class HybridRetriever:
    def __init__(
        self,
        qdrant: QdrantClient,
        collection_name: str,
        encoder: RegulationEncoder,
        reranker: Reranker,
        initial_k: int = 20,
        reranker_top_n: int = 5,
    ) -> None:
        self.qdrant = qdrant
        self.collection_name = collection_name
        self.encoder = encoder
        self.reranker = reranker
        self.initial_k = initial_k
        self.reranker_top_n = reranker_top_n
        self._sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")

    def _encode_sparse(self, text: str) -> SparseVector:
        result = list(self._sparse_model.query_embed(text))[0]
        return SparseVector(
            indices=result.indices.tolist(),
            values=result.values.tolist(),
        )

    def search(
        self,
        query: str,
        qdrant_filter: Filter | None = None,
        expand_to_parent: bool = True,
        top_n: int | None = None,
    ) -> list[dict]:
        """
        Retrieve and rerank relevant sections for a query.

        Returns a list of dicts, each representing one unique section with:
          - section_number, section_title, section_node
          - full_text (all chunks concatenated, for generation context)
          - score (from reranker)
          - metadata (full payload from best matching chunk)
        """
        # Encode query
        dense_vec = self.encoder.encode_query(query)
        sparse_vec = self._encode_sparse(query)

        # Hybrid search via Qdrant native RRF
        prefetch_limit = self.initial_k * 2
        response = self.qdrant.query_points(
            collection_name=self.collection_name,
            prefetch=[
                Prefetch(
                    query=dense_vec,
                    using="",
                    limit=prefetch_limit,
                    filter=qdrant_filter,
                ),
                Prefetch(
                    query=sparse_vec,
                    using="bm25",
                    limit=prefetch_limit,
                    filter=qdrant_filter,
                ),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=self.initial_k,
            with_payload=True,
        )
        candidates = response.points

        if not candidates:
            return []

        # Rerank candidates
        reranked = self.reranker.rerank(query, candidates, top_n=top_n or self.reranker_top_n)

        if not expand_to_parent:
            return [
                {
                    "section_number": c.payload.get("section_number", ""),
                    "section_title": c.payload.get("section_title", ""),
                    "section_node": c.payload.get("section_node", ""),
                    "full_text": c.payload.get("chunk_text", ""),
                    "score": c.score,
                    "metadata": c.payload,
                }
                for c in reranked
            ]

        # Parent expansion: fetch full section for each unique section_node
        results: list[dict] = []
        seen_nodes: set[str] = set()

        for chunk in reranked:
            node = chunk.payload.get("section_node", "")
            if node in seen_nodes:
                continue
            seen_nodes.add(node)

            siblings = scroll_by_node(self.qdrant, self.collection_name, node)
            full_text = "\n\n".join(
                s.payload.get("chunk_text", "") for s in siblings
            )

            results.append({
                "section_number": chunk.payload.get("section_number", ""),
                "section_title": chunk.payload.get("section_title", ""),
                "section_node": node,
                "full_text": full_text,
                "score": chunk.score,
                "metadata": chunk.payload,
            })

        return results
