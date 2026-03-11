# CLAUDE.md — PhFlow Project Context

## Project Overview
RAG application for FDA Title 21 CFR regulatory intelligence. Multi-purpose: Q&A, compliance checking, research synthesis. Fully local stack — no external APIs.

## Build & Run Commands

```bash
# Install
pip install -e ".[ingestion,api]"    # ingestion + API
pip install -e ".[dev]"              # add test/lint tools
pip install -e ".[redis]"            # Redis session persistence (optional)

# GPU-accelerated FAISS (via conda, not pip)
conda install -c pytorch faiss-gpu

# Infrastructure (llama.cpp only — no separate vector DB server needed)
llama-server -m /path/to/model.gguf -c 65536 --port 8080

# Ingestion (one-time, ~25 min on CPU)
python scripts/ingest.py --reset
python scripts/verify_collection.py  # spot-check

# API
uvicorn phflow.api.main:app --reload --reload-dir src --reload-dir ui --port 8000

# Tests
pytest tests/unit/
pytest tests/eval/ -m integration -v   # requires running FAISS + llama.cpp
ruff check src/
```

## Architecture

```
src/phflow/
  settings.py               # Pydantic BaseSettings, reads .env
  ingestion/
    parser.py               # lxml.iterparse streaming XML (never loads full DOM)
    cleaner.py              # text extraction, TABLE → markdown
    chunker.py              # parent-child chunking, 512-token max
    metadata.py             # versions-title-21.json enrichment
    pipeline.py             # orchestrates parse → chunk → embed → upsert
  embeddings/encoder.py     # BAAI/bge-base-en-v1.5, 768-dim, LRU query cache
  vectorstore/
    client.py               # FaissStore: IndexFlatIP + metadata.json + bm25_sparse.json
    schema.py               # create_index(), index_exists()
  retrieval/
    hybrid.py               # dense FAISS + BM25 → client-side RRF → parent expand
    filters.py              # filter predicate builder + auto filter extraction
  generation/
    llm.py                  # LlamaCppClient — POST /completion, SSE streaming
    prompts.py              # prompt assembly, query rewrite, adaptive history window
    citations.py            # CFR context block + source summary formatting
    chain.py                # retrieve → RRF → generate (full RAG orchestrator)
  api/
    main.py                 # FastAPI app factory with lifespan
    schemas.py              # Pydantic request/response models
    dependencies.py         # FastAPI Depends injectors
    session_store.py        # SessionStore abstraction (in-memory + Redis)
    routers/health.py       # GET /health
    routers/search.py       # POST /api/v1/search, GET /api/v1/sections/{node}, GET /api/v1/parts
    routers/chat.py         # POST /api/v1/chat, POST /api/v1/chat/{id}/message
scripts/
  ingest.py                 # CLI: --reset, --batch-size
  verify_collection.py      # spot-check FAISS payloads
prompts/
  system_prompt.txt         # loaded at runtime by prompts.py
ui/
  index.html                # single-page Q&A app (served at /)
  static/app.js
  static/style.css
tests/unit/                 # test_chunker.py, test_cleaner.py
tests/eval/                 # golden_set.json + retrieval quality tests (P@5, R@5)
```

## Key Technical Decisions

- **Streaming XML parser**: `lxml.iterparse` with `elem.clear()` — 21.7MB file, never load full DOM
- **Chunking**: Parent-child model. Full section in `chunk_text` payload for parent expansion. Child chunks embedded at 512-token max, paragraph label boundaries
- **Vector store**: FAISS `IndexFlatIP` (exact search, no approximation). On-disk: `dense.index` + `metadata.json` + `bm25_sparse.json`. GPU auto-detected, falls back to CPU
- **Hybrid retrieval**: Client-side RRF over FAISS dense + fastembed BM25 sparse (`Qdrant/bm25` model). No reranker
- **No reranker**: Dropped. RRF fusion of dense + sparse is sufficient; reranker added latency without measurable gain on regulatory text
- **Default scope**: When no explicit `part_number` is specified, retrieval is scoped to `SCOPE_PART_NUMBERS=[210, 211]`. Explicit part filter overrides scope
- **LLM**: llama.cpp HTTP server at `localhost:8080`. Client in `generation/llm.py`. Non-streaming: `POST /completion`. Streaming: SSE lines prefixed `"data: "`
- **Query rewrite**: Enabled by default (`ENABLE_QUERY_REWRITE=true`). Short LLM call (128 tokens) maps operational language → CFR terminology before retrieval encode
- **History window**: 0 turns by default (prevents hallucination on fresh queries). Expands to 2 turns when query contains back-reference pronouns
- **Session state**: In-memory dict (default). Set `REDIS_URL` for Redis-backed persistence with 24-hour TTL
- **System prompt**: Loaded from `prompts/system_prompt.txt` at startup. Edit the file and restart to change prompt behaviour
- **Answer guards**: Multiple post-generation checks — bare header echo, too-short, prompt echo, template placeholder, `<think>` leak

## Data

- XML: `data/ecfr-title-21/ECFR-title21.xml` — 244K lines, ~16,828 DIV8 sections
- Versions: `data/ecfr-title-21/versions-title-21.json` — joined on `identifier == section_number.replace("§ ", "").strip()`
- FAISS index: `faiss_index/` — ~50K chunks, 768-dim dense + BM25 sparse sidecar

## Environment Variables (.env)

```
EMBEDDING_MODEL=BAAI/bge-base-en-v1.5
EMBEDDING_DEVICE=cuda
EMBEDDING_BATCH_SIZE=64
FAISS_INDEX_DIR=faiss_index
LLAMA_CPP_URL=http://localhost:8080
LLAMA_CPP_TEMPERATURE=0.1
LLAMA_CPP_NUM_CTX=65536
LLAMA_CPP_MAX_TOKENS=768
INITIAL_RETRIEVAL_K=10
RETRIEVAL_TOP_N=5
PREFETCH_MULTIPLIER=2
ENABLE_QUERY_REWRITE=true
CHUNK_MAX_TOKENS=512
XML_PATH=data/ecfr-title-21/ECFR-title21.xml
VERSIONS_PATH=data/ecfr-title-21/versions-title-21.json
# Optional:
# REDIS_URL=redis://localhost:6379
# SCOPE_PART_NUMBERS=[210,211]
```

## Common Patterns

### Adding a new API endpoint
1. Add Pydantic models to `api/schemas.py`
2. Create router function in `api/routers/`
3. Register with `app.include_router(...)` in `api/main.py`
4. Use `Depends(get_retriever)` or `Depends(get_chain)` from `api/dependencies.py`

### Changing the LLM
Edit `generation/llm.py` — the `LlamaCppClient` interface (`generate()`, `stream()`, `correct_query()`, `health()`, `close()`) is the contract consumed by `chain.py`.

### Changing the embedding model
Update `EMBEDDING_MODEL` in `.env`. Re-run `ingest.py --reset` (index dimensions must match model output).

### Adding metadata filters
Add a new field to `build_filter()` in `retrieval/filters.py`, expose it via `SearchFilters` in `api/schemas.py`, and pass it through `chain.py`'s `RAGChain.run()` signature.

### Changing the system prompt
Edit `prompts/system_prompt.txt` directly and restart the API. The file is loaded once at startup via `prompts.py`.
