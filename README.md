# PhFlow: FDA Regulatory Intelligence

A fully local RAG application for asking natural-language questions about **Title 21 CFR (Food and Drugs)**,
focused on pharmaceutical purchase order compliance. Answers are grounded in and cite specific CFR sections.
No external APIs.

## How it works

```
Query → auto-filter → LLM query rewrite → multi-query decomposition
      → per sub-query: BGE dense + BM25 sparse → RRF fusion
      → union + dedup by section → parent expansion → LLM → answer guards
```

- **Structure-aware ingestion:** streams the 21.7 MB eCFR XML (~16.8K sections) into ~50K parent-child chunks,
  converting tables to markdown.
- **Hybrid retrieval:** FAISS dense search (`bge-base-en-v1.5`) plus BM25, fused with Reciprocal Rank Fusion.
  RRF matched reranker quality without the 300 to 800 ms latency.
- **Query rewrite:** maps operational language ("what do I do if...") to CFR terminology before retrieval.
- **Multi-query decomposition:** splits multi-part questions into 2 to 3 sub-queries that retrieve independently.
- **Answer guards:** catch empty, echoed or leaked-reasoning answers and fall back to a safe response.

Design rationale: [DESIGN.md](DESIGN.md).

## Tech stack

FastAPI, FAISS, `BAAI/bge-base-en-v1.5`, fastembed BM25, llama.cpp or Ollama, vanilla JS UI, optional Redis sessions.

## Setup

Requires Python 3.11+ and either Ollama or llama.cpp.

```bash
pip install -e ".[ingestion,api]"
cp .env.example .env
```

**Start an LLM** (pick one):

```bash
ollama pull qwen3.5                                         # .env: LLM_BACKEND=ollama, LLAMA_CPP_URL=http://localhost:11434
llama-server -m /path/to/model.gguf -c 65536 --port 8080    # .env: LLM_BACKEND=llamacpp
```

**Add data:** download the Title 21 XML and versions JSON from the [eCFR bulk data site](https://www.ecfr.gov/current/title-21)
into `data/ecfr-title-21/`, then build the index (~25 min on CPU, ~4 min on GPU):

```bash
python scripts/ingest.py --reset
python scripts/verify_collection.py
```

**Run:**

```bash
uvicorn phflow.api.main:app --host 0.0.0.0 --port 8000
```

Open http://localhost:8000 for the Q&A UI, or http://localhost:8000/docs for the API.

## API

| Endpoint | Description |
|---|---|
| `POST /api/v1/chat` | Ask a question (supports `filters`, `stream`) |
| `POST /api/v1/chat/{session_id}/message` | Continue a session |
| `POST /api/v1/search` | Retrieval only |
| `GET /api/v1/sections/{section_node}`, `GET /api/v1/parts` | Section lookup |
| `GET /health` | Component health |

```json
POST /api/v1/chat
{ "message": "What are the GMP requirements for drug product labeling?", "filters": {"part_number": 211} }
```

## Configuration

All settings live in `.env` (see `.env.example`). The most useful:

| Variable | Default | Description |
|---|---|---|
| `LLM_BACKEND` | `llamacpp` | `llamacpp` or `ollama` |
| `EMBEDDING_DEVICE` | `cuda` | `cpu`, `cuda` or `mps` (falls back to CPU) |
| `RETRIEVAL_TOP_N` | `10` | Sections passed to the LLM |
| `ENABLE_QUERY_REWRITE` | `true` | Rewrite queries into CFR terms |
| `SCOPE_PART_NUMBERS` | `[210, 211]` | Default CFR parts when no filter is given |
| `REDIS_URL` | unset | Enables persistent sessions |

## Development

```bash
pip install -e ".[dev]"
pytest tests/unit/
pytest tests/eval/ -m integration -v   # retrieval quality (P@5, R@5); needs the index and LLM running
ruff check src/
```
