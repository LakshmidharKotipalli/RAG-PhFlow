"""
FastAPI application factory with lifespan-managed component initialization.

All heavy components (encoder, Qdrant client, llama.cpp client) are
initialized once at startup and shared via app.state.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

# Absolute path to ui/ — safe regardless of working directory
_UI_DIR = Path(__file__).parents[3] / "ui"

from phflow.api.routers import chat, health, search
from phflow.api.session_store import make_session_store
from phflow.settings import settings
from phflow.embeddings.encoder import RegulationEncoder
from phflow.generation.chain import RAGChain
from phflow.generation.llm import LlamaCppClient
from phflow.retrieval.hybrid import HybridRetriever

from phflow.vectorstore.client import get_qdrant_client

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing PhFlow components...")

    # Embedding encoder
    encoder = RegulationEncoder(
        model_name=settings.embedding_model,
        device=settings.embedding_device,
        batch_size=settings.embedding_batch_size,
    )
    logger.info("Encoder loaded: %s (dim=%d)", settings.embedding_model, encoder.embedding_dim)



    # Qdrant
    qdrant = get_qdrant_client()
    logger.info("Qdrant connected: %s", settings.qdrant_url)

    # llama.cpp server
    llm = LlamaCppClient()
    if await llm.health():
        logger.info("llama.cpp client ready @ %s", settings.llama_cpp_url)
    else:
        logger.warning(
            "llama.cpp server not reachable at %s — start it before sending queries",
            settings.llama_cpp_url,
        )

    # Retriever
    retriever = HybridRetriever(
        qdrant=qdrant,
        collection_name=settings.collection_name,
        encoder=encoder,
    )

    # RAG chain
    chain = RAGChain(retriever=retriever, llm=llm)

    # Session store (Redis if REDIS_URL is set, else in-memory)
    session_store = make_session_store(settings.redis_url)
    backend = "Redis" if hasattr(session_store, "_redis") else "in-memory"
    logger.info("Session store: %s", backend)

    # Store in app state for dependency injection
    app.state.encoder = encoder

    app.state.qdrant = qdrant
    app.state.llm = llm
    app.state.retriever = retriever
    app.state.chain = chain
    app.state.session_store = session_store

    logger.info("All components initialized. API ready.")
    yield

    # Cleanup
    await llm.close()
    logger.info("PhFlow shutdown complete.")


def create_app() -> FastAPI:
    app = FastAPI(
        title="PhFlow — FDA Regulatory Intelligence API",
        description=(
            "Multi-purpose RAG application for Title 21 CFR (Food and Drugs). "
            "Supports Q&A, compliance checking, and regulatory research."
        ),
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type", "Authorization", "X-Session-Id"],
    )

    app.include_router(health.router)
    app.include_router(search.router)
    app.include_router(chat.router)

    app.mount("/static", StaticFiles(directory=str(_UI_DIR / "static")), name="static")

    @app.get("/", include_in_schema=False)
    async def serve_ui():
        return FileResponse(str(_UI_DIR / "index.html"))

    return app


app = create_app()
