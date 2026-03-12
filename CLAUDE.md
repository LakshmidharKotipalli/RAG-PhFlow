# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RAG application for FDA Title 21 CFR regulatory intelligence. Multi-purpose: Q&A, compliance checking, research synthesis. Fully local stack — no external APIs. See `DESIGN.md` for detailed pipeline rationale.

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

# API (development)
uvicorn phflow.api.main:app --reload --reload-dir src --reload-dir ui --port 8000

# Tests
pytest tests/unit/                          # unit tests only
pytest tests/unit/test_chunker.py           # single test file
pytest tests/unit/test_chunker.py -k "test_name"  # single test function
pytest tests/eval/ -m integration -v        # requires running FAISS + llama.cpp

# Lint
ruff check src/
```

**Python**: 3.11+ required. **Ruff**: line-length 100, target py311. **Pytest**: `asyncio_mode = "auto"`.

## Architecture

All source code lives under `src/phflow/`. The pipeline flows:

```
XML → parse → clean → chunk → embed → FAISS index
Query → [auto-filter] → [query rewrite] → [multi-query decomposition] → dense + BM25 → RRF fusion → union + dedup → parent expand → LLM → answer guards → response
```

**Key modules and their roles:**

- `settings.py` — Pydantic BaseSettings, reads `.env`
- `ingestion/` — Streaming XML parse (`lxml.iterparse`, never full DOM) → text cleaning (TABLE→markdown) → parent-child structural chunking (512-token max) → embed → upsert
- `embeddings/encoder.py` — BAAI/bge-base-en-v1.5 (768-dim). Asymmetric encoding: queries get instruction prefix, passages don't. LRU query cache
- `vectorstore/client.py` — `FaissStore`: FAISS `IndexFlatIP` (exact search) + on-disk `metadata.json` + `bm25_sparse.json`. GPU auto-detected
- `retrieval/hybrid.py` — Dense FAISS + BM25 sparse → client-side RRF (k=60) → parent expansion (reassemble full sections from child chunks)
- `retrieval/filters.py` — Post-FAISS metadata filtering + auto-extraction of part/chapter/date from query text
- `generation/llm.py` — `LlamaCppClient` supports both llama.cpp (`/completion`) and Ollama (`/api/generate`) backends. Interface: `generate()`, `stream()`, `correct_query()`, `health()`, `close()`. Backend selected via `LLM_BACKEND` env var
- `generation/prompts.py` — Prompt assembly, query rewrite (operational language → CFR terms), multi-query decomposition, adaptive history window (0 turns default, 2 on back-references). System prompt hot-reloaded from disk on every query
- `generation/citations.py` — CFR context block formatting + `SOURCES_USED:` parsing
- `generation/chain.py` — Full RAG orchestrator: rewrite → decompose → retrieve per sub-query → union + dedup → generate → answer guards
- `api/` — FastAPI app with routers (`chat.py`, `search.py`, `health.py`), `dependencies.py` for DI, `session_store.py` (in-memory or Redis)
- `prompts/system_prompt.txt` — System prompt hot-reloaded on every query. Edit the file and the next request picks up the change — no restart needed

## Key Technical Decisions

- **Streaming XML**: `lxml.iterparse` + `elem.clear()` — 21.7MB file, peak memory <200MB
- **Chunking**: Parent-child model. Children embedded at 512-token max (paragraph label boundaries). Parent text stored but not embedded — used for parent expansion after retrieval
- **Vector store**: FAISS `IndexFlatIP` (exact search). At ~50K vectors, exact search is fast enough; no ANN approximation needed. On-disk: `dense.index` + `metadata.json` + `bm25_sparse.json`
- **Hybrid retrieval**: Client-side RRF over dense + BM25 sparse (fastembed `Qdrant/bm25`). No reranker — RRF achieves equivalent P@5/R@5 without 300-800ms reranker latency
- **Multi-query decomposition**: Complex queries are decomposed into 2-3 sub-queries via a short LLM call. Each sub-query retrieves independently, results are unioned and deduplicated by section_number (best RRF score wins). Solves the single-vector-direction problem where one query can't cover multiple regulatory concepts simultaneously
- **LLM backend**: Supports both llama.cpp (native `/completion`) and Ollama (`/api/generate`). Set `LLM_BACKEND=ollama` and `OLLAMA_MODEL=qwen3.5` in `.env`. Ollama requires `think: false` for reasoning models like Qwen3
- **Default scope**: Retrieval scoped to `SCOPE_PART_NUMBERS=[210, 211]` when no explicit part filter. Explicit filter overrides
- **Query rewrite**: Enabled by default. Short LLM call (128 tokens) maps operational language → CFR terminology before retrieval encode. Includes action→requirement flipping ("what do I do" → "regulatory requirements for") and violation→deviation mapping ("not in storage" → "deviation from storage requirements"). Original question preserved for prompt assembly
- **History window**: 0 turns default (prevents hallucination). Expands to 2 turns when query contains back-reference pronouns (`"that"`, `"it"`, `"the section"`, etc.)
- **Answer guards**: Post-generation checks in `chain.py` — bare header echo, too-short (<60 chars), prompt echo, template placeholder, `<think>` leak. Any failure → standard insufficient-context response
- **Session state**: In-memory dict default. `REDIS_URL` env var enables Redis with 24-hour TTL

## Data

- XML source: `data/ecfr-title-21/ECFR-title21.xml` — 244K lines, ~16,828 DIV8 sections
- Versions: `data/ecfr-title-21/versions-title-21.json` — amendment history, joined on `identifier == section_number.replace("§ ", "").strip()`
- FAISS index: `faiss_index/` — ~50K chunks, 768-dim dense + BM25 sparse sidecar
- Debug logs: `query_logs/` — per-query JSON files with full pipeline trace (gitignored)

## Environment Variables (.env)

```
EMBEDDING_MODEL=BAAI/bge-base-en-v1.5
EMBEDDING_DEVICE=cuda              # cpu, cuda, or mps (auto-fallback)
EMBEDDING_BATCH_SIZE=64
FAISS_INDEX_DIR=faiss_index
LLM_BACKEND=llamacpp             # "llamacpp" or "ollama"
OLLAMA_MODEL=qwen3.5              # only used when LLM_BACKEND=ollama
LLAMA_CPP_URL=http://localhost:8080  # port 8080 for llama.cpp, 11434 for ollama
LLAMA_CPP_TEMPERATURE=0.1
LLAMA_CPP_NUM_CTX=65536            # must match -c on llama-server
LLAMA_CPP_MAX_TOKENS=768
INITIAL_RETRIEVAL_K=20
RETRIEVAL_TOP_N=10
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
Set `LLM_BACKEND=ollama` and `OLLAMA_MODEL=<model>` in `.env` for Ollama, or `LLM_BACKEND=llamacpp` for llama.cpp. The `LlamaCppClient` in `generation/llm.py` handles both backends. Interface: `generate()`, `stream()`, `correct_query()`, `health()`, `close()`.

### Changing the embedding model
Update `EMBEDDING_MODEL` in `.env`. Re-run `ingest.py --reset` (index dimensions must match model output).

### Adding metadata filters
Add a new field to `build_filter()` in `retrieval/filters.py`, expose it via `SearchFilters` in `api/schemas.py`, and pass it through `chain.py`'s `RAGChain.run()` signature.

### Changing the system prompt
Edit `prompts/system_prompt.txt` directly. The file is hot-reloaded on every query — no restart needed.