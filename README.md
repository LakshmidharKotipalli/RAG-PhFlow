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
│                          (text chunks only)        └──► Qdrant  │
│                                     Qwen3-Reranker-0.6B ──► top-5│
│                                     Parent expansion ──► LLM   │
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
├── Makefile                      # dev/run/ingest shortcuts
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
│   │   ├── reranker.py           # Qwen3-Reranker-0.6B (yes/no logit scoring)
│   │   └── filters.py            # Qdrant Filter builder
│   ├── generation/
│   │   ├── llm.py                # llama.cpp HTTP client
│   │   ├── prompts.py            # QA system prompt + query builder
│   │   ├── citations.py          # context block + source summary formatting
│   │   └── chain.py              # full RAG pipeline orchestrator
│   └── api/
│       ├── main.py               # FastAPI app factory + static file serving
│       ├── schemas.py            # Pydantic request/response models
│       ├── dependencies.py       # FastAPI Depends injectors
│       └── routers/              # health.py, search.py, chat.py
│
├── scripts/
│   ├── ingest.py                 # ingestion CLI
│   └── verify_collection.py     # spot-check Qdrant after ingest
├── tests/
│   ├── unit/                     # test_chunker.py, test_cleaner.py
│   └── integration/
├── ui/
│   ├── index.html                # single-page Q&A app
│   └── static/
│       ├── style.css
│       └── app.js
└── query_logs/                   # per-query pipeline debug logs (gitignored)
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
make qdrant
# or: docker compose up -d
```

Qdrant dashboard: http://localhost:6333/dashboard

### 3. Start llama.cpp server

```bash
llama-server -m /path/to/Llama-3.1-8B-Instruct-Q4_K_M.gguf \
             -c 65536 \
             --port 8080

# Verify it's running:
curl http://localhost:8080/health
```

Recommended models (from [bartowski on HuggingFace](https://huggingface.co/bartowski)):
- `Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf` — best balance of speed/quality
- `Mistral-7B-Instruct-v0.3-Q4_K_M.gguf` — faster, lower memory
- `Meta-Llama-3.1-70B-Instruct-Q4_K_M.gguf` — highest quality (requires ~40 GB RAM)

> **Context window:** The server must be started with `-c 65536` to match `LLAMA_CPP_NUM_CTX=65536` in `.env`. Using a smaller value silently caps the context.

### 4. Configure environment

```bash
cp .env.example .env
# Edit .env as needed
```

### 5. Run ingestion

```bash
make ingest
# or: python scripts/ingest.py --reset

# Verify:
make verify
```

Estimated time: ~25 minutes on CPU, ~4 minutes with GPU.

### 6. Start the API + UI

```bash
make run          # production (no reload)
make dev          # development (auto-reload on src/ and ui/ changes only)
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

## RAG Pipeline

### Retrieval

```
User query
  │
  ├─► PubMedBERT encode ──────────────────────────────────┐
  │                                                        │
  └─► fastembed BM25 encode ─────────────────────────────►│
                              (text chunks only)           │
                                              Qdrant RRF Fusion
                                              (top-80 dense + top-80 sparse)
                                                           │
                                                 top-40 candidates
                                                           │
                                           Qwen3-Reranker-0.6B
                                           (yes/no logit scoring)
                                                           │
                                                     top-5 sections
                                                           │
                                             Parent expansion
                                             (fetch full section text)
                                                           │
                                                 LLM generation
```

### Generation

The LLM receives:
1. **System prompt** — role definition and strict answering rules (formal English, minimum 3 sentences, no symbols, cite only from context)
2. **Conversation history** — last 1 turn for follow-up context
3. **Retrieved context** — up to 5 sections labelled `[SOURCE 1]`…`[SOURCE N]`
4. **Question** — the user's query

The model must declare which sources it used at the end of its answer (`SOURCES_USED: 1, 3`). Only those sources are shown in the UI.

### Per-query debug logs

Every query writes a timestamped log to `query_logs/` capturing:
- Original query
- Retrieval candidates (pre-rerank) with RRF scores
- Reranked results with actual reranker probabilities
- Full assembled prompt
- Raw LLM output
- Final parsed answer and declared sources

---

## Configuration

All settings are in `.env` (see `.env.example`):

| Variable | Default | Description |
|---|---|---|
| `EMBEDDING_MODEL` | `NeuML/pubmedbert-base-embeddings` | Sentence-transformer model |
| `EMBEDDING_DEVICE` | `cuda` | `cpu`, `cuda`, or `mps` |
| `QDRANT_URL` | `http://localhost:6333` | Qdrant server URL |
| `COLLECTION_NAME` | `ecfr_title21` | Qdrant collection |
| `LLAMA_CPP_URL` | `http://localhost:8080` | llama.cpp server URL |
| `LLAMA_CPP_TEMPERATURE` | `0.1` | LLM temperature (low = factual) |
| `LLAMA_CPP_NUM_CTX` | `65536` | Context window — must match `-c` on llama-server |
| `LLAMA_CPP_MAX_TOKENS` | `2048` | Max response tokens |
| `RERANKER_MODEL` | `Qwen/Qwen3-Reranker-0.6B` | Cross-encoder reranker |
| `INITIAL_RETRIEVAL_K` | `40` | Candidates before reranking |
| `RERANKER_TOP_N` | `5` | Sections passed to LLM |
| `RERANKER_SCORE_THRESHOLD` | `0.0` | Min reranker probability (0 = no filter) |
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
