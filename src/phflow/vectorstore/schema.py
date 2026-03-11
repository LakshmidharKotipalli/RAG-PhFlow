"""
FAISS index creation and management.

Replaces the Qdrant collection schema.  FAISS uses IndexFlatIP for
exhaustive inner-product search (equivalent to cosine for L2-normalised
vectors).  Metadata filtering is handled in Python at query time.
"""

from __future__ import annotations

from phflow.vectorstore.client import FaissStore


def create_index(store: FaissStore) -> None:
    """Reset and initialise an empty FAISS index."""
    store.reset()


def index_exists(store: FaissStore) -> bool:
    """Check whether a persisted FAISS index exists on disk."""
    return store.exists()
