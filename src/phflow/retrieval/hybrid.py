"""
Hybrid search: dense + BM25 sparse retrieval fused with Reciprocal Rank Fusion (RRF)
via Qdrant's native hybrid query API.

Flow:
  1. Encode query -> dense vector (BGE, with instruction prefix) + sparse vector (BM25)
  2. Qdrant RRF fusion -> top-K candidates
  3. Parent expansion -> full section text for each top-N result
"""

from __future__ import annotations

from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import (
    FieldCondition,
    Filter,
    Fusion,
    FusionQuery,
    MatchValue,
    Prefetch,
    SparseVector,
)

from phflow.settings import settings
from phflow.embeddings.encoder import RegulationEncoder
from phflow.vectorstore.client import scroll_by_node


# Exclude authority chunks (amendment history, statutory citations) which are
# generic boilerplate and dominate BM25 for unrelated queries.
_EXCLUDE_AUTHORITY = Filter(
    must_not=[FieldCondition(key="chunk_type", match=MatchValue(value="authority"))]
)


def _merge_filters(base: Filter | None, extra: Filter) -> Filter:
    """Combine two Qdrant filters by AND-ing their must conditions."""
    must = list(extra.must or [])
    if base and base.must:
        must.extend(base.must)
    return Filter(must=must)


class HybridRetriever:
    def __init__(
        self,
        qdrant: QdrantClient,
        collection_name: str,
        encoder: RegulationEncoder,
        initial_k: int | None = None,
        top_n: int | None = None,
    ) -> None:
        self.qdrant = qdrant
        self.collection_name = collection_name
        self.encoder = encoder
        self.initial_k = initial_k or settings.initial_retrieval_k
        self.top_n = top_n or settings.retrieval_top_n
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
        search_filter = _merge_filters(qdrant_filter, _EXCLUDE_AUTHORITY)

        # Hybrid search via Qdrant native RRF
        prefetch_limit = self.initial_k * settings.prefetch_multiplier
        response = self.qdrant.query_points(
            collection_name=self.collection_name,
            prefetch=[
                Prefetch(
                    query=dense_vec,
                    using="",
                    limit=prefetch_limit,
                    filter=search_filter,
                ),
                Prefetch(
                    query=sparse_vec,
                    using="bm25",
                    limit=prefetch_limit,
                    filter=search_filter,
                ),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=self.initial_k,
            with_payload=True,
        )
        candidates = response.points

        if not candidates:
            if logger:
                logger.log("RETRIEVAL", "No candidates returned from Qdrant.")
            return []

        # Take top-N by RRF score
        top_candidates = candidates[:effective_top_n]

        if logger:
            logger.log("RETRIEVAL (RRF top-N)", [
                {
                    "section_number": c.payload.get("section_number", ""),
                    "section_title": c.payload.get("section_title", ""),
                    "rrf_score": round(c.score, 6),
                }
                for c in top_candidates
            ])

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
                for c in top_candidates
            ]

        # Parent expansion: fetch full section for each unique section_node
        results: list[dict] = []
        seen_nodes: set[str] = set()

        for chunk in top_candidates:
            node = chunk.payload.get("section_node", "")
            if node in seen_nodes:
                continue
            seen_nodes.add(node)

            siblings = scroll_by_node(self.qdrant, self.collection_name, node)

            # Concatenate sibling chunks into one full section text.
            # Keep the header from the first chunk only; strip it from
            # subsequent chunks so the same header does not repeat.
            texts: list[str] = []
            for i, s in enumerate(siblings):
                text = s.payload.get("chunk_text", "")
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
                "section_number": chunk.payload.get("section_number", ""),
                "section_title": chunk.payload.get("section_title", ""),
                "section_node": node,
                "full_text": full_text,
                "score": chunk.score,
                "metadata": chunk.payload,
            })

        return results
