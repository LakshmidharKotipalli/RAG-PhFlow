# CLAUDE.md — PhFlow Project Context

## Project Overview
RAG application for FDA Title 21 CFR regulatory intelligence. Multi-purpose: Q&A, compliance checking, research synthesis. Fully local stack — no external APIs.

## Build & Run Commands

```bash
# Install
pip install -e ".[ingestion,api]"    # ingestion + API
pip install -e ".[ui]"               # add Chainlit UI
pip install -e ".[dev]"              # add test/lint tools

# Infrastructure
docker compose up -d                 # start Qdrant (port 6333)
llama-server -m /path/to/model.gguf -c 32768 --port 8080   # start llama.cpp

# Ingestion (one-time, ~25 min on CPU)
python scripts/ingest.py --reset
python scripts/verify_collection.py  # spot-check

# API
uvicorn phflow.api.main:app --reload --port 8000

# UI
chainlit run ui/app.py

# Tests
pytest tests/unit/
pytest tests/
ruff check src/
```

## Architecture

```
src/phflow/
  config/settings.py        # Pydantic BaseSettings, reads .env
  ingestion/
    parser.py               # lxml.iterparse streaming XML (never loads full DOM)
    cleaner.py              # text extraction, TABLE → markdown
    chunker.py              # parent-child chunking, 400-token target
    metadata.py             # versions-title-21.json enrichment
    pipeline.py             # orchestrates parse → chunk → embed → upsert
  embeddings/encoder.py     # NeuML/pubmedbert-base-embeddings, 768-dim
  vectorstore/
    client.py               # Qdrant client + upsert helpers
    schema.py               # collection config + payload indexes
  retrieval/
    hybrid.py               # dense + BM25 RRF → Qwen3 rerank → parent expand
    reranker.py             # Qwen/Qwen3-Reranker-0.6B (yes/no logit scoring)
    filters.py              # Qdrant Filter builder
  generation/
    llm.py                  # LlamaCppClient — POST /completion, SSE streaming
    prompts.py              # QA / compliance / research prompt templates
    citations.py            # CFR citation formatting
    chain.py                # retrieve → rerank → generate
  api/
    main.py                 # FastAPI app factory with lifespan
    schemas.py              # Pydantic request/response models
    dependencies.py         # FastAPI Depends injectors
    routers/health.py       # GET /health
    routers/search.py       # POST /api/v1/search, GET /api/v1/sections/{node}
    routers/chat.py         # POST /api/v1/chat, POST /api/v1/chat/{id}/message
    routers/compliance.py   # POST /api/v1/compliance/check
scripts/
  ingest.py                 # CLI: --reset, --batch-size
  verify_collection.py      # spot-check Qdrant payloads
ui/app.py                   # Chainlit frontend
tests/unit/                 # test_chunker.py, test_cleaner.py
```

## Key Technical Decisions

- **Streaming XML parser**: `lxml.iterparse` with `elem.clear()` — 21.7MB file, never load full DOM
- **Chunking**: Parent-child model. Full section in `parent_text` payload (not embedded). Child chunks embedded at 400-token target, paragraph boundaries
- **Hybrid retrieval**: Qdrant native RRF (`Prefetch` dense + `Prefetch` BM25 → `FusionQuery`)
- **Reranker**: `Qwen3-Reranker-0.6B` scores via yes/no token logits (not standard cross-encoder)
- **LLM**: llama.cpp HTTP server at `localhost:8080`. Client in `generation/llm.py`. Non-streaming: `POST /completion`. Streaming: SSE lines prefixed `"data: "`
- **Session state**: In-memory dict in `routers/chat.py`. Swap `_sessions` for Redis in production
- **Metadata filtering**: Qdrant `Filter` built from query params — `part_number`, `chapter_number`, `amended_after`, `chunk_type`, `is_removed=False` (default)

## Data

- XML: `data/ecfr-title-21/ECFR-title21.xml` — 244K lines, ~16,828 DIV8 sections
- Versions: `data/ecfr-title-21/versions-title-21.json` — joined on `identifier == section_number.replace("§ ", "").strip()`
- Qdrant collection: `ecfr_title21`, ~50K chunks, 768-dim dense + BM25 sparse

## Environment Variables (.env)

```
EMBEDDING_MODEL=NeuML/pubmedbert-base-embeddings
EMBEDDING_DEVICE=cpu
QDRANT_URL=http://localhost:6333
COLLECTION_NAME=ecfr_title21
LLAMA_CPP_URL=http://localhost:8080
LLAMA_CPP_TEMPERATURE=0.1
LLAMA_CPP_NUM_CTX=32768
LLAMA_CPP_MAX_TOKENS=2048
RERANKER_MODEL=Qwen/Qwen3-Reranker-0.6B
INITIAL_RETRIEVAL_K=20
RERANKER_TOP_N=5
CHUNK_TARGET_TOKENS=400
CHUNK_MAX_TOKENS=512
```

## Common Patterns

### Adding a new API endpoint
1. Add Pydantic models to `api/schemas.py`
2. Create router function in `api/routers/`
3. Register with `app.include_router(...)` in `api/main.py`
4. Use `Depends(get_retriever)` or `Depends(get_chain)` from `api/dependencies.py`

### Changing the LLM
Edit `generation/llm.py` — the `LlamaCppClient` interface (`generate()`, `stream()`, `health()`, `close()`) is the contract consumed by `chain.py`.

### Changing the embedding model
Update `EMBEDDING_MODEL` in `.env`. Re-run `ingest.py --reset` (collection dim must match).

### Adding metadata filters
Add a new field to `build_filter()` in `retrieval/filters.py`, then expose it via `SearchFilters` in `api/schemas.py`.
