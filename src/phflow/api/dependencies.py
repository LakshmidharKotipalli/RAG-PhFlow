"""
FastAPI dependency injectors.
Reads initialized components from app.state (set in lifespan).
"""

from __future__ import annotations

from fastapi import Request

from phflow.api.session_store import SessionStore
from phflow.generation.chain import RAGChain
from phflow.retrieval.hybrid import HybridRetriever


def get_retriever(request: Request) -> HybridRetriever:
    return request.app.state.retriever


def get_chain(request: Request) -> RAGChain:
    return request.app.state.chain


def get_session_store(request: Request) -> SessionStore:
    return request.app.state.session_store
