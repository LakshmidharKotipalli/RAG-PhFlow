"""
Qdrant collection schema and payload index configuration.

Collection: ecfr_title21
  - Dense vectors: 768-dim cosine, HNSW m=16 ef_construct=200
  - Sparse vectors: BM25 (fastembed Qdrant/bm25)
  - Payload: rich regulatory metadata with indexed fields for filtered retrieval
"""

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    HnswConfigDiff,
    PayloadSchemaType,
    SparseIndexParams,
    SparseVectorParams,
    VectorParams,
)


def create_collection(client: QdrantClient, collection_name: str, vector_size: int = 768) -> None:
    """Create the Qdrant collection with dense + sparse vector config."""
    client.recreate_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=vector_size,
            distance=Distance.COSINE,
            hnsw_config=HnswConfigDiff(m=16, ef_construct=200),
        ),
        sparse_vectors_config={
            "bm25": SparseVectorParams(
                index=SparseIndexParams(on_disk=True)
            )
        },
    )
    _create_payload_indexes(client, collection_name)


def _create_payload_indexes(client: QdrantClient, collection_name: str) -> None:
    """Create payload indexes for fast metadata filtering."""
    integer_fields = ["part_number", "title_number", "chunk_index", "total_chunks", "token_count"]
    keyword_fields = [
        "section_number", "section_node", "section_title",
        "chapter_number", "subchapter_letter", "subpart_letter",
        "chunk_type", "latest_amendment_date", "issue_date",
    ]
    bool_fields = ["is_removed", "has_table", "substantive_change"]

    for field in integer_fields:
        client.create_payload_index(collection_name, field, PayloadSchemaType.INTEGER, timeout=60)

    for field in keyword_fields:
        client.create_payload_index(collection_name, field, PayloadSchemaType.KEYWORD, timeout=60)

    for field in bool_fields:
        client.create_payload_index(collection_name, field, PayloadSchemaType.BOOL, timeout=60)


def collection_exists(client: QdrantClient, collection_name: str) -> bool:
    return client.collection_exists(collection_name)
