"""
Qdrant client factory and upsert utilities.
"""

from __future__ import annotations

import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, SparseVector

from phflow.config.settings import settings


def get_qdrant_client() -> QdrantClient:
    return QdrantClient(url=settings.qdrant_url, timeout=60)


def build_point(
    chunk_text: str,
    dense_vector: list[float],
    sparse_vector: SparseVector,
    payload: dict,
) -> PointStruct:
    """Construct a single Qdrant PointStruct from chunk data."""
    return PointStruct(
        id=str(uuid.uuid4()),
        vector={
            "": dense_vector,       # default dense vector
            "bm25": sparse_vector,  # sparse BM25 vector
        },
        payload={**payload, "chunk_text": chunk_text},
    )


def upsert_batch(
    client: QdrantClient,
    collection_name: str,
    points: list[PointStruct],
) -> None:
    """Upsert a batch of points into the collection."""
    client.upsert(collection_name=collection_name, points=points, wait=True)


def scroll_by_node(
    client: QdrantClient,
    collection_name: str,
    section_node: str,
    limit: int = 100,
) -> list:
    """Fetch all chunks belonging to a section node, ordered by chunk_index."""
    from qdrant_client.models import FieldCondition, Filter, MatchValue

    results, _ = client.scroll(
        collection_name=collection_name,
        scroll_filter=Filter(
            must=[FieldCondition(key="section_node", match=MatchValue(value=section_node))]
        ),
        with_payload=True,
        with_vectors=False,
        limit=limit,
    )
    results.sort(key=lambda p: p.payload.get("chunk_index", 0))
    return results
