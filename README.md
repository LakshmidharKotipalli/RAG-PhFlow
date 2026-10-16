# PhFlow — FDA Regulatory Intelligence

A RAG (Retrieval-Augmented Generation) application for querying and reasoning over **Title 21 of the Code of Federal Regulations (FDA — Food and Drugs)**, focused on pharmaceutical and medical purchase order compliance.

Ask natural-language questions and get synthesised, well-explained answers grounded in specific CFR sections — fully local, no external APIs required.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│              Browser  (http://localhost:8000)                    │
│              Vanilla JS single-page Q&A interface               │
└────────────────────────────┬────────────────────────────────────┘
                             │ HTTP / JSON
┌────────────────────────────▼────────────────────────────────────┐
│                     FastAPI (port 8000)                         │
│              /chat   /search   /sections   /health              │
└───┬──────────────────────────────────────────────────────┬──────┘
    │                                                      │
    ▼                                                      ▼
┌───────────────────────────────┐          ┌──────────────────────┐
│        RAG Chain              │          │  LLM Backend         │
│  rewrite → decompose →        │          │  llama.cpp or Ollama │
│  retrieve → fuse → generate   │          │  (any GGUF/Ollama    │
└───┬───────────────────────────┘          │   model)             │
    │                                      └──────────────────────┘
    ▼
┌──────────────────────────────────────────────────────────────────┐
│                    Hybrid Retrieval                              │
│                                                                  │
│  Query ──► auto-extract filters (Part/Chapter/date hints)        │
│        ──► LLM query rewrite → formal CFR terms                  │
│        ──► LLM multi-query decomposition → 2-3 sub-queries       │
│            ┌──────────────────────────────────────────┐          │
│            │  Per sub-query:                          │          │
│            │  BGE embed (w/ prefix) ──► Dense  ──┐   │          │
│            │  BM25 (fastembed)      ──► Sparse ──┤RRF│          │
│            └──────────────────────────────────┘  └───┘          │
│        ──► Union + dedup by section (best score wins)            │
│        ──► Parent expansion ──► LLM                              │
└──────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│   FAISS Index (Local Disk)      │
│   ~50,000 structural chunks     │
│   Dense 768-dim + BM25 sparse   │
└─────────────────────────────────┘
```

### Tech Stack

| Component | Technology |
|---|---|
| UI | Vanilla HTML/CSS/JS (served by FastAPI) |
| Embedding | `BAAI/bge-base-en-v1.5` (768-dim, asymmetric query prefix, LRU-cached) |
| Sparse search | `fastembed` — `Qdrant/bm25` |
| Vector DB | FAISS (IndexFlatIP, on-disk metadata + BM25 sidecar) |
| Fusion | Client-side Reciprocal Rank Fusion (RRF, k=60) |
| LLM | llama.cpp or Ollama (configurable via `LLM_BACKEND`) |
| Query decomposition | LLM-based multi-query split for multi-faceted questions |
| API | FastAPI + uvicorn |
| Session store | In-memory dict (default) or Redis (opt-in via `REDIS_URL`) |

---

## Data

Source: `data/ecfr-title-21/ECFR-title21.xml`
- 21.7 MB, 244,264 lines
- ~16,828 regulatory sections (DIV8 elements), ~1,145 embedded tables
- Hierarchy: Title → Chapter → Subchapter → Part → Subpart → Section
- Supporting file: `data/ecfr-title-21/versions-title-21.json` (amendment history)

> **Note:** The XML and JSON data files are not included in this repository due to size. Download them from the [eCFR bulk data site](https://www.ecfr.gov/current/title-21) and place them in `data/ecfr-title-21/`.

---

## Project Structure

```
RAG-PhFlow/
├── pyproject.toml
├── .env                          # local config (copy from .env.example)
├── prompts/system_prompt.txt     # system prompt loaded at runtime
├── data/ecfr-title-21/           # source XML + metadata (not in repo)
├── src/phflow/
│   ├── settings.py               # Pydantic BaseSettings
│   ├── ingestion/
│   │   ├── parser.py             # lxml.iterparse streaming XML parser
│   │   ├── cleaner.py            # text extraction, table → markdown
│   │   ├── chunker.py            # structural chunker (one chunk per top-level CFR paragraph)
│   │   ├── metadata.py           # versions.json enrichment
│   │   └── pipeline.py           # ingestion orchestrator
│   ├── embeddings/encoder.py     # BGE sentence-transformer wrapper + LRU query cache
│   ├── vectorstore/
│   │   ├── client.py             # FAISS client + upsert helpers
│   │   └── schema.py             # index creation helpers
│   ├── retrieval/
│   │   ├── hybrid.py             # dense + sparse RRF → parent expand
│   │   └── filters.py            # filter predicate builder + auto filter extraction
│   ├── generation/
│   │   ├── llm.py                # LLM client (llama.cpp + Ollama backends)
│   │   ├── prompts.py            # system prompt (hot-reload), query rewrite, multi-query decomposition
│   │   ├── citations.py          # context block + source summary formatting
│   │   └── chain.py              # full RAG pipeline: rewrite → decompose → retrieve → generate
│   └── api/
│       ├── main.py               # FastAPI app factory + startup lifespan
│       ├── schemas.py            # Pydantic request/response models
│       ├── dependencies.py       # FastAPI Depends injectors
│       ├── session_store.py      # SessionStore abstraction (in-memory + Redis)
│       └── routers/              # health.py, search.py, chat.py
│
├── scripts/
│   ├── ingest.py                 # ingestion CLI
│   └── verify_collection.py     # spot-check FAISS after ingest
├── tests/
│   ├── unit/                     # test_chunker.py, test_cleaner.py
│   ├── integration/
│   └── eval/                     # golden_set.json + retrieval quality tests (P@5, R@5)
├── ui/
│   ├── index.html                # single-page Q&A app
│   └── static/
│       ├── style.css
│       └── app.js
└── query_logs/                   # per-query pipeline debug logs — JSON only (gitignored)
```

---

## Setup

### Prerequisites

- Python 3.11+
- llama.cpp installed (`brew install llama.cpp` on macOS, or build from source)
- A GGUF model file (e.g. `Llama-3.1-8B-Instruct-Q4_K_M.gguf`)

### 1. Install dependencies

```bash
pip install -e ".[ingestion,api]"

# Optional: Redis-backed session persistence
pip install -e ".[redis]"

# Optional: GPU-accelerated FAISS (via conda, not pip)
conda install -c pytorch faiss-gpu
```

### 2. FAISS Index
FAISS is fully local and runs inside the Python process. No separate vector database server is required. During ingestion, the index is built and saved locally to `faiss_index/`.

### 3. Start LLM server

**Option A: Ollama (recommended for quick setup)**

```bash
ollama pull qwen3.5
# Verify:
curl http://localhost:11434/api/tags
```

Set in `.env`:
```
LLM_BACKEND=ollama
OLLAMA_MODEL=qwen3.5
LLAMA_CPP_URL=http://localhost:11434
```

**Option B: llama.cpp (for GGUF models)**

```bash
llama-server -m /path/to/Llama-3.1-8B-Instruct-Q4_K_M.gguf \
             -c 65536 \
             --port 8080

# Verify:
curl http://localhost:8080/health
```

Set in `.env`:
```
LLM_BACKEND=llamacpp
LLAMA_CPP_URL=http://localhost:8080
```

Recommended GGUF models (from [bartowski on HuggingFace](https://huggingface.co/bartowski)):
- `Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf` — best balance of speed/quality
- `Mistral-7B-Instruct-v0.3-Q4_K_M.gguf` — faster, lower memory
- `Meta-Llama-3.1-70B-Instruct-Q4_K_M.gguf` — highest quality (requires ~40 GB RAM)

> **Context window:** For llama.cpp, start with `-c 65536` to match `LLAMA_CPP_NUM_CTX=65536` in `.env`.

> **Startup health check:** The API warns in the log if the LLM server is unreachable at startup — it does not hard-fail so you can start the server after the API if needed.

### 4. Configure environment

```bash
cp .env.example .env
# Edit .env as needed
```

### 5. Run ingestion

```bash
python scripts/ingest.py --reset

# Verify:
python scripts/verify_collection.py
```

Estimated time: ~25 minutes on CPU, ~4 minutes with GPU.

### 6. Start the API + UI

```bash
# Production (no reload)
uvicorn phflow.api.main:app --host 0.0.0.0 --port 8000

# Development (auto-reload on src/ and ui/ changes only)
uvicorn phflow.api.main:app --reload --reload-dir src --reload-dir ui --host 0.0.0.0 --port 8000
```

Open **http://localhost:8000** — the Q&A interface loads in your browser.

API docs (Swagger): http://localhost:8000/docs

---

## API Reference

### Health

```
GET /health
```
```json
{
  "status": "ok",
  "components": {
    "faiss": {"status": "ok"},
    "llama_cpp": {"status": "ok"},
    "encoder": {"status": "ok"}
  }
}
```

### Chat (Q&A)

```
POST /api/v1/chat
```
```json
{
  "message": "What are the GMP requirements for drug product labeling?",
  "mode": "qa",
  "filters": {"part_number": 211},
  "stream": false
}
```

Response:
```json
{
  "session_id": "uuid",
  "answer": "Under 21 CFR Part 211, drug product labels must include...",
  "sources": [
    {
      "section_number": "§ 211.68",
      "section_title": "Automatic, mechanical, and electronic equipment.",
      "full_text": "..."
    }
  ],
  "mode": "qa"
}
```

**Streaming:** Set `"stream": true`. Response is `text/plain` token stream with `X-Session-Id` header for session continuation.

**Continue a session:**
```
POST /api/v1/chat/{session_id}/message
```

### Search (retrieval only)

```
POST /api/v1/search
```
```json
{
  "query": "labeling requirements for dietary supplements",
  "filters": {"part_number": 101},
  "top_k": 5,
  "expand_to_parent": true
}
```

### Section Lookup

```
GET /api/v1/sections/{section_node}
GET /api/v1/parts
```

---

## Configuration

All settings are in `.env` (see `.env.example`):

| Variable | Default | Description |
|---|---|---|
| `EMBEDDING_MODEL` | `BAAI/bge-base-en-v1.5` | Sentence-transformer model |
| `EMBEDDING_DEVICE` | `cuda` | `cpu`, `cuda`, or `mps` (auto-falls back to cpu) |
| `EMBEDDING_BATCH_SIZE` | `64` | Batch size for ingestion encoding |
| `FAISS_INDEX_DIR` | `faiss_index` | Directory for FAISS index files |
| `LLM_BACKEND` | `llamacpp` | `"llamacpp"` or `"ollama"` |
| `OLLAMA_MODEL` | `qwen3.5` | Ollama model name (only when `LLM_BACKEND=ollama`) |
| `LLAMA_CPP_URL` | `http://localhost:8080` | LLM server URL (8080 for llama.cpp, 11434 for Ollama) |
| `LLAMA_CPP_TEMPERATURE` | `0.1` | LLM temperature (low = factual) |
| `LLAMA_CPP_NUM_CTX` | `65536` | Context window — must match `-c` on llama-server |
| `LLAMA_CPP_MAX_TOKENS` | `768` | Max response tokens |
| `INITIAL_RETRIEVAL_K` | `20` | Candidates after RRF fusion |
| `RETRIEVAL_TOP_N` | `10` | Sections passed to LLM |
| `PREFETCH_MULTIPLIER` | `2` | Over-fetch factor per search branch before RRF |
| `ENABLE_QUERY_REWRITE` | `true` | Rewrite colloquial queries to CFR terminology before retrieval |
| `SCOPE_PART_NUMBERS` | `[210, 211]` | Default CFR parts to scope retrieval when no explicit part filter |
| `REDIS_URL` | _(unset)_ | Redis URL for persistent sessions (e.g. `redis://localhost:6379`) |
| `CHUNK_MAX_TOKENS` | `512` | Overflow guard for structural chunking |
| `XML_PATH` | `data/ecfr-title-21/ECFR-title21.xml` | Source XML location |
| `VERSIONS_PATH` | `data/ecfr-title-21/versions-title-21.json` | Amendment versions JSON |

---

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Optional: Redis session persistence
pip install -e ".[redis]"

# Run unit tests
pytest tests/unit/

# Run retrieval quality evaluation (requires FAISS index + models running)
pytest tests/eval/ -m integration -v

# Run all tests (requires FAISS + llama.cpp running)
pytest tests/

# Lint
ruff check src/
```

---
