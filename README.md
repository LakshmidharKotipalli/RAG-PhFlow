# PhFlow — FDA Regulatory Intelligence

A RAG (Retrieval-Augmented Generation) application for querying and reasoning over **Title 21 of the Code of Federal Regulations (FDA — Food and Drugs)**.

Ask natural-language questions and get answers cited to specific CFR sections — fully local, no external APIs required.

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
│   /chat   /search   /compliance/check   /sections   /health     │
└───┬──────────────────────────────────────────────────────┬──────┘
    │                                                      │
    ▼                                                      ▼
┌───────────────────────────────┐          ┌──────────────────────┐
│        RAG Chain              │          │  llama.cpp server    │
│  retrieve → rerank → generate │          │  port 8080           │
└───┬───────────────────────────┘          │  (any GGUF model)    │
    │                                      └──────────────────────┘
    ▼
┌──────────────────────────────────────────────────────────────────┐
│                    Hybrid Retrieval                              │
│                                                                  │
│  Query ──► PubMedBERT embed ──► Dense vector ─────┐             │
│        ──► BM25 (fastembed)  ──► Sparse vector ───┤  RRF Fusion │
│                                                   └──► Qdrant   │
│                                     Qwen3-Reranker-0.6B ──► top-5│
│                                     Parent expansion ──► LLM    │
└──────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│   Qdrant  (port 6333)           │
│   Collection: ecfr_title21      │
│   ~50,000 chunks (DIV8 sections)│
│   Dense 768-dim + BM25 sparse   │
└─────────────────────────────────┘
```

### Tech Stack

| Component | Technology |
|---|---|
| UI | Vanilla HTML/CSS/JS (served by FastAPI) |
| Embedding | `NeuML/pubmedbert-base-embeddings` (768-dim) |
| Sparse search | `fastembed` — `Qdrant/bm25` |
| Vector DB | Qdrant (hybrid dense + sparse, HNSW) |
| Reranker | `Qwen/Qwen3-Reranker-0.6B` |
| LLM | llama.cpp server (any GGUF model) |
| API | FastAPI + uvicorn |

---

## Data

Source: `data/ecfr-title-21/ECFR-title21.xml`
- 21.7 MB, 244,264 lines
- ~16,828 regulatory sections (DIV8 elements)
- Hierarchy: Title → Chapter → Subchapter → Part → Subpart → Section
- Supporting file: `data/ecfr-title-21/versions-title-21.json` (amendment history)

> **Note:** The XML and JSON data files are not included in this repository due to size. Download them from the [eCFR bulk data site](https://www.ecfr.gov/current/title-21) and place them in `data/ecfr-title-21/`.

---

## Project Structure

```
RAG-PhFlow/
├── pyproject.toml
├── .env                          # local config (copy from .env.example)
├── docker-compose.yml            # Qdrant service
├── data/ecfr-title-21/           # source XML + metadata (not in repo)
│
├── src/phflow/
│   ├── config/settings.py        # Pydantic BaseSettings
│   ├── ingestion/
│   │   ├── parser.py             # lxml.iterparse streaming XML parser
│   │   ├── cleaner.py            # text extraction, table → markdown
│   │   ├── chunker.py            # paragraph-boundary chunker (parent-child model)
│   │   ├── metadata.py           # versions.json enrichment
│   │   └── pipeline.py           # ingestion orchestrator
│   ├── embeddings/encoder.py     # PubMedBERT sentence-transformer wrapper
│   ├── vectorstore/
│   │   ├── client.py             # Qdrant client + upsert helpers
│   │   └── schema.py             # collection + payload index setup
│   ├── retrieval/
│   │   ├── hybrid.py             # dense + sparse RRF → rerank → parent expand
│   │   ├── reranker.py           # Qwen3-Reranker-0.6B
│   │   └── filters.py            # Qdrant Filter builder
│   ├── generation/
│   │   ├── llm.py                # llama.cpp HTTP client (streaming + non-streaming)
│   │   ├── prompts.py            # QA / compliance / research prompt templates
│   │   ├── citations.py          # CFR citation formatting
│   │   └── chain.py              # RAG chain
│   └── api/
│       ├── main.py               # FastAPI app factory + lifespan + static file serving
│       ├── schemas.py            # Pydantic request/response models
│       ├── dependencies.py       # FastAPI Depends injectors
│       └── routers/              # health.py, search.py, chat.py, compliance.py
│
├── scripts/
│   ├── ingest.py                 # ingestion CLI
│   └── verify_collection.py     # spot-check Qdrant after ingest
├── tests/
│   ├── unit/                     # test_chunker.py, test_cleaner.py
│   └── integration/
└── ui/
    ├── index.html                # single-page Q&A app
    └── static/
        ├── style.css
        └── app.js
```

---

## Setup

### Prerequisites

- Python 3.11+
- Docker (for Qdrant)
- llama.cpp installed (`brew install llama.cpp` on macOS, or build from source)
- A GGUF model file (e.g. `Llama-3.1-8B-Instruct-Q4_K_M.gguf`)

### 1. Install dependencies

```bash
pip install -e ".[ingestion,api]"
```

### 2. Start Qdrant

```bash
docker compose up -d
```

Qdrant dashboard: http://localhost:6333/dashboard

### 3. Start llama.cpp server

```bash
llama-server -m /path/to/Llama-3.1-8B-Instruct-Q4_K_M.gguf \
             -c 32768 \
             --port 8080

# Verify it's running:
curl http://localhost:8080/health
```

Recommended models (from [bartowski on HuggingFace](https://huggingface.co/bartowski)):
- `Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf` — best balance of speed/quality
- `Mistral-7B-Instruct-v0.3-Q4_K_M.gguf` — faster, lower memory
- `Meta-Llama-3.1-70B-Instruct-Q4_K_M.gguf` — highest quality (requires ~40 GB RAM)

### 4. Configure environment

```bash
cp .env.example .env
# Edit .env — set LLAMA_CPP_URL if your server runs on a different port
```

### 5. Run ingestion

```bash
# First time — creates collection and indexes ~50K chunks (~25 min on CPU):
python scripts/ingest.py --reset

# Verify:
python scripts/verify_collection.py
```

### 6. Start the API + UI

```bash
uvicorn phflow.api.main:app --reload --port 8000
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
    "qdrant": {"status": "ok"},
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
  "message": "What are the GMPs for food manufacturing?",
  "mode": "qa",
  "filters": {"part_number": 110},
  "stream": false
}
```

Response:
```json
{
  "session_id": "uuid",
  "answer": "...",
  "sources": [{"section_number": "§ 110.10", "section_title": "...", "full_text": "..."}],
  "citations_footer": "...",
  "mode": "qa"
}
```

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

### Compliance Check

```
POST /api/v1/compliance/check
```
```json
{
  "scenario": "We manufacture a dietary supplement in capsule form and our label shows a Supplement Facts panel but omits the serving size declaration.",
  "filters": {"part_number": 101}
}
```

### Section Lookup

```
GET /api/v1/sections/{section_node}
GET /api/v1/parts
```

---

## Chunking Strategy

Each DIV8 section is split into **parent + child chunks**:

- **Parent text** — full section stored in Qdrant payload (`parent_text`), NOT embedded. Used for LLM context after retrieval.
- **Child chunks** — embedded into the vector index:
  - Target: 400 tokens, max 512 tokens (matches PubMedBERT context window)
  - Split at `<P>` paragraph boundaries first, then sentence boundaries
  - Short sections (<150 tokens) → single chunk
  - Tables → `chunk_type="table"`
  - AUTH/SOURCE blocks → `chunk_type="authority"`

Every chunk is prefixed with a regulatory context header:
```
21 CFR § 1.24 — Exemptions from required label statements.
Part 1: General Enforcement Regulations | Subpart B: General Labeling Requirements

<chunk text>
```

---

## Retrieval Pipeline

```
Query
  │
  ├─► PubMedBERT encode ──────────────────────────────────┐
  │                                                        │
  └─► fastembed BM25 encode ─────────────────────────────►│
                                                           │
                                              Qdrant RRF Fusion
                                              (top-40 dense + top-40 sparse)
                                                           │
                                                    top-20 candidates
                                                           │
                                            Qwen3-Reranker-0.6B
                                                           │
                                                      top-5 sections
                                                           │
                                              Parent expansion
                                              (fetch full section text)
                                                           │
                                                  LLM generation
```

---

## Configuration

All settings are in `.env` (see `.env.example`):

| Variable | Default | Description |
|---|---|---|
| `EMBEDDING_MODEL` | `NeuML/pubmedbert-base-embeddings` | Sentence-transformer model |
| `EMBEDDING_DEVICE` | `cpu` | `cpu`, `cuda`, or `mps` |
| `QDRANT_URL` | `http://localhost:6333` | Qdrant server URL |
| `COLLECTION_NAME` | `ecfr_title21` | Qdrant collection |
| `LLAMA_CPP_URL` | `http://localhost:8080` | llama.cpp server URL |
| `LLAMA_CPP_TEMPERATURE` | `0.1` | LLM temperature (low = factual) |
| `LLAMA_CPP_NUM_CTX` | `32768` | Context window (tokens) |
| `LLAMA_CPP_MAX_TOKENS` | `2048` | Max response tokens |
| `RERANKER_MODEL` | `Qwen/Qwen3-Reranker-0.6B` | Cross-encoder reranker |
| `INITIAL_RETRIEVAL_K` | `20` | Candidates before reranking |
| `RERANKER_TOP_N` | `5` | Sections passed to LLM |
| `CHUNK_TARGET_TOKENS` | `400` | Target chunk size |
| `CHUNK_MAX_TOKENS` | `512` | Maximum chunk size |

---

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run unit tests
pytest tests/unit/

# Run all tests (requires Qdrant + llama.cpp running)
pytest tests/

# Lint
ruff check src/
```

---

## License

This project is for research and educational use. The regulatory data (ECFR Title 21) is published by the U.S. Government Printing Office and is in the public domain.
