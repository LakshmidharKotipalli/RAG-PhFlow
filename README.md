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
│  retrieve → fuse → generate   │          │  port 8080           │
└───┬───────────────────────────┘          │  (any GGUF model)    │
    │                                      └──────────────────────┘
    ▼
┌──────────────────────────────────────────────────────────────────┐
│                    Hybrid Retrieval                              │
│                                                                  │
│  Query ──► auto-extract filters (Part/Chapter/date hints)        │
│        ──► [optional] LLM query rewrite → formal CFR terms       │
│        ──► BGE embed (w/ prefix) ──► Dense vector ────┐         │
│        ──► BM25 (fastembed)      ──► Sparse vector ───┤  RRF    │
│                       (text + table chunks)            └►top-N  │
│                                     Parent expansion ──► LLM    │
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
| LLM | llama.cpp server (any GGUF model) |
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
│   │   ├── llm.py                # llama.cpp HTTP client
│   │   ├── prompts.py            # QA system prompt, query rewrite, adaptive history window
│   │   ├── citations.py          # context block + source summary formatting
│   │   └── chain.py              # full RAG pipeline orchestrator
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

> **Startup health check:** The API warns in the log if llama.cpp is unreachable at startup — it does not hard-fail so you can start the server after the API if needed.

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

## Pipeline — Detailed Design

This section documents every design decision in the pipeline from raw XML to final answer.

### 1. XML Parsing

**File:** `src/phflow/ingestion/parser.py`

The source data is a single 21.7 MB XML file with 244,264 lines. Loading the full DOM would require ~1 GB of memory, so the parser uses `lxml.iterparse` in streaming mode:

- Events are consumed one `end` event at a time
- `elem.clear()` is called immediately after processing each `DIV8` (section) to release the subtree from memory
- `huge_tree=True` is passed to handle deeply nested elements without hitting lxml's default size limits
- `recover=True` tolerates minor XML encoding issues in the source file

**Hierarchy walked:**

```
DIV1 (Title 21)
  └─ DIV3 (Chapter)
      └─ DIV4 (Subchapter)
          └─ DIV5 (Part)
              └─ DIV6 (Subpart)
                  └─ DIV8 (Section)  ← one SectionData emitted per DIV8
```

An ancestor stack tracks the current Chapter → Part → Subpart hierarchy so each section carries full location metadata without re-scanning the file.

---

### 2. Text Cleaning

**File:** `src/phflow/ingestion/cleaner.py`

Each `DIV8` section element is converted to plain text before chunking:

- **Inline tags** (`<I>`, `<E>`, `<SU>`, `<BCAP>`, etc.) are accumulated and joined in-line — they carry emphasis or superscripts but do not break paragraph flow
- **Skip tags** (`<AMDDATE>`, `<EDNOTE>`, `<EFFDNOT>`, etc.) are silently dropped — they contain editorial or amendment metadata not relevant to regulatory content
- **`<TABLE>` elements** are converted to pipe-delimited markdown (`| col | col |`) so their content is searchable and readable by the LLM
- **`<AUTH>` / `<SOURCE>` blocks** are captured separately and tagged `[AUTH]` / `[SOURCE]` in the paragraph list so the chunker can route them to authority-type chunks
- Whitespace is normalised: consecutive spaces collapsed, runs of 3+ newlines reduced to 2

---

### 3. Chunking Strategy

**File:** `src/phflow/ingestion/chunker.py`

**Model: Structural parent-child chunking**

The core insight is that CFR sections are hierarchically labelled — `(a)`, `(b)`, `(c)` at the top level, with `(1)`, `(2)`, `(i)`, `(ii)`, `(A)` nested beneath. This structure maps directly to a chunking boundary without needing any token-count heuristic.

**Parent text (stored, not embedded)**

The full concatenation of all paragraphs in the section is stored in the FAISS payload as `parent_text`. It is never embedded. It exists so that, after retrieval identifies the relevant section via its children, the full section text (not just the matching chunk) is fetched and passed to the LLM.

**Child chunks (embedded)**

One child chunk is created per top-level paragraph group:

| Group type | Trigger | `chunk_type` |
|---|---|---|
| Intro | Unlabelled text before the first `(a)` | `"text"` |
| Paragraph group | Each `(a)`, `(b)`, `(c)`, … group + all its sub-paragraphs | `"text"` |
| Table | Any `<TABLE>` element (converted to markdown) | `"table"` |
| Authority | `<AUTH>` / `<SOURCE>` blocks | `"authority"` |

**Label detection and roman numeral disambiguation**

The chunker must distinguish `(i)` as a roman numeral sub-label from `(i)` as a standalone top-level label (rare but present). The rule:

- If a single-character label is in `{i, v, x, l, c, d, m}` AND a prior top-level letter has already been seen in this section, treat it as a sub-label, not a new top-level group
- `(a)` always resets the top-level sequence (some sections repeat the alphabet for different regulatory topics)
- Multi-character lowercase (e.g. `ii`, `iii`) are always roman numerals → sub-labels

**Overflow splitting**

If a paragraph group exceeds `max_tokens` (512, matching the embedding model's context limit), it is split at `(1)`, `(2)`, … number-label boundaries. This preserves the structure of long lists (common in Part 201 labelling requirements) without cutting mid-sentence.

**Chunk header**

Every chunk — text, table, or authority — is prefixed with a context header:

```
21 CFR § 211.182 — § 211.182 Equipment cleaning and use log.
Part 211: Current Good Manufacturing Practice for Finished Pharmaceuticals | Subpart J: Records and Reports
```

This header ensures that even when the embedding model sees only a fragment of the section, it receives the full regulatory address, improving retrieval precision.

**Authority chunks excluded from retrieval**

`chunk_type="authority"` chunks (amendment citations, statutory authority) are filtered out at query time. They contain boilerplate like `"Authority: 21 U.S.C. 321, 351…"` which scores high on BM25 for almost any regulatory query and dominates results. Only `text` and `table` chunks are searched.

**Chunk statistics**

~50,000–55,000 total chunks across ~16,828 sections. Average: 3–4 text/table chunks per section.

---

### 4. Embedding

**File:** `src/phflow/embeddings/encoder.py`

**Model:** `BAAI/bge-base-en-v1.5`
- 768-dimensional dense vectors
- 512-token context window (matches `chunk_max_tokens`)
- Cosine similarity space
- Strong BEIR benchmark performance on factual/regulatory retrieval

**Asymmetric encoding**

BGE-base uses an asymmetric retrieval design: queries and passages are encoded differently to align their vector spaces:

- **Passages (chunks):** encoded with `model.encode(text)` — no prefix
- **Queries:** encoded with `model.encode("Represent this sentence for searching relevant passages: " + query)` — the instruction prefix shifts the query vector toward the passage space

Without this prefix, query vectors and chunk vectors occupy different regions of the 768-dim space, degrading retrieval recall by 5–15% on typical BEIR tasks.

Both query and passage vectors are L2-normalised before storage, so cosine similarity reduces to dot product.

**Query vector cache**

`encode_query()` maintains a module-level LRU cache (256 entries). Repeated or near-identical queries — common during a session — skip model inference entirely. The model is deterministic, so cached results are always correct.

---

### 5. Indexing

**File:** `src/phflow/vectorstore/client.py`, `src/phflow/vectorstore/schema.py`

**Vector index — Dense (FAISS IndexFlatIP)**

```
Distance:       Inner product (equivalent to cosine for L2-normalised vectors)
Dimensions:     768
GPU support:    Auto-detected (faiss-gpu via conda); falls back to CPU
```

Vectors are L2-normalised at ingestion time, so inner-product search is equivalent to cosine similarity. `IndexFlatIP` provides exact nearest-neighbour search — no approximation error.

**Index file layout**

```
faiss_index/
├── dense.index       # FAISS binary (always saved as CPU-compatible)
├── metadata.json     # list[dict] — payload for each vector, aligned by position
└── bm25_sparse.json  # sparse BM25 vectors: list[{"indices": [...], "values": [...]}]
```

**Payload fields**

Every chunk stores a rich metadata payload. Key fields used for filtering:

| Field | Type | Use |
|---|---|---|
| `part_number` | Integer | Filter to a specific CFR Part (e.g. `part_number=211`) |
| `title_number` | Integer | Always 21 for this corpus |
| `chunk_index` | Integer | Position within section |
| `total_chunks` | Integer | Total chunks in parent section |
| `token_count` | Integer | Chunk size in embedding-model tokens |
| `section_number` | Keyword | e.g. `"§ 211.182"` |
| `section_node` | Keyword | e.g. `"21:4.0.1.1.11.10.6.2"` — unique section address used for parent expansion |
| `section_title` | Keyword | Section heading text |
| `chapter_number` | Keyword | Roman numeral chapter (e.g. `"I"`) |
| `subchapter_letter` | Keyword | Subchapter letter |
| `subpart_letter` | Keyword | Subpart letter (e.g. `"B"`) |
| `chunk_type` | Keyword | `"text"`, `"table"`, `"authority"` |
| `latest_amendment_date` | Keyword | ISO date — used for `amended_after` temporal filter |
| `issue_date` | Keyword | Federal Register issue date |
| `is_removed` | Bool | True if section was repealed — always filtered out at query time |
| `has_table` | Bool | True if section contains a table |
| `substantive_change` | Bool | From versions metadata — marks substantive vs. editorial amendments |

---

### 6. Retrieval

**File:** `src/phflow/retrieval/hybrid.py`

#### 6.1 Auto Filter Extraction

**File:** `src/phflow/retrieval/filters.py` → `extract_filters_from_query()`

Before retrieval, the raw query is scanned for explicit filter hints and any detected values are merged with caller-supplied filters (caller values always take precedence):

| Pattern in query | Auto-set filter | Example |
|---|---|---|
| `"Part 211"` / `"part 211"` | `part_number=211` | "What are Part 211 GMP requirements?" |
| `"§ 211.42"` | `part_number=211` | "What does § 211.42 require?" |
| `"Chapter I"` | `chapter_number="I"` | "Chapter I labeling rules" |
| `"amended after 2022"` | `amended_after="2022-01-01"` | "rules amended after 2022" |

This improves precision for queries that name a specific Part without the user setting a filter in the API request.

#### 6.2 Default Part Scope

When no explicit `part_number` filter is present in the query or request, the retriever automatically scopes to `SCOPE_PART_NUMBERS` (default: Parts 210 and 211). This ensures that generic queries about pharmaceutical GMP stay within the core manufacturing regulations without accidentally pulling from unrelated titles.

Setting an explicit `part_number` in the request (or query text) overrides the default scope entirely.

#### 6.3 Optional Query Rewrite

**File:** `src/phflow/generation/prompts.py` → `build_retrieval_rewrite_prompt()`

When `ENABLE_QUERY_REWRITE=true` (the default), a short LLM call (`max_tokens=128`) rewrites colloquial workplace language into formal CFR terminology before the retrieval encode step. For example:

- `"fridge temperature for drugs"` → `"storage temperature requirements for pharmaceutical drug products"`
- `"sign off on a digital form"` → `"electronic record signature requirements"`
- `"pallet rejected at dock"` → `"container receipt and acceptance refusal"`

The rewrite only affects the retrieval query. The original question is still shown to the user and used for prompt assembly.

#### 6.4 Hybrid Search (Dense + BM25 via RRF)

Both retrieval signals are computed and fused client-side using Reciprocal Rank Fusion (RRF):

```
Query
  │
  ├─► encode_query(prefix + query)  →  dense vector
  │                                         │
  └─► bm25.query_embed(query)        →  sparse vector
                                            │
              Client-side RRF fusion
              ┌────────────────────────────────────────┐
              │  FAISS search:  dense top-20            │
              │  BM25 dot-product: sparse top-20        │
              │                                        │
              │  RRF merge  →  top-10 candidates       │
              └────────────────────────────────────────┘
```

**Why RRF instead of score interpolation?**

Dense and BM25 scores exist on incompatible scales. RRF avoids the need to tune a weighting factor `α` by working purely on rank positions:

```
RRF(d) = Σ  1 / (k + rank_i(d))
```

with `k=60`. A chunk ranked 1st in both lists gets a higher RRF score than one ranked 5th in both, regardless of the raw score values. This is robust to scale differences and avoids expensive calibration.

**Authority chunk exclusion**

A permanent filter is AND-ed into every search that excludes `chunk_type="authority"` chunks. These statutory citation blocks score high on BM25 for almost any regulatory query (they contain part numbers, U.S.C. citations) but carry no substantive content.

**Prefetch limits**

Each search (dense and sparse) fetches `initial_k × PREFETCH_MULTIPLIER` candidates (default `2`, so 20 each). After RRF merge, the top `initial_k = 10` are passed forward. The multiplier is configurable via `PREFETCH_MULTIPLIER` in `.env` to tune the recall/latency tradeoff.

#### 6.5 Metadata Filtering

Filters are implemented as Python callables applied post-FAISS (over-fetch then filter):

| Filter | Field | Example |
|---|---|---|
| CFR Part | `part_number` (int) | `part_number=211` |
| Part scope | `part_numbers` (list) | `part_numbers=[210, 211]` |
| Chapter | `chapter_number` (keyword) | `chapter_number="I"` |
| Subpart | `subpart_letter` (keyword) | `subpart_letter="B"` |
| Chunk type | `chunk_type` (keyword) | `chunk_types=["text","table"]` |
| Amendment date | `latest_amendment_date` (range) | `amended_after="2020-01-01"` |
| Has table | `has_table` (bool) | `has_table=True` |
| Exclude repealed | `is_removed` (bool) | always `is_removed=False` |

The `is_removed=False` filter is always active to exclude repealed sections.

---

### 7. Parent Expansion

**File:** `src/phflow/retrieval/hybrid.py` → `HybridRetriever.search()`

After RRF fusion, the top-N results are individual *chunks* (one paragraph group each). The LLM needs the *full section* for context — a single `(b)` sub-paragraph without its surrounding `(a)` and `(c)` may be ambiguous or incomplete.

For each unique `section_node` in the top-N results:

1. Scroll all chunks sharing that `section_node` from FAISS metadata
2. Concatenate their `chunk_text` in `chunk_index` order
3. Strip the repeated section header from subsequent chunks (only the first chunk's header is kept)
4. Replace the single matched chunk with the full assembled section text

This ensures the LLM always receives complete regulatory sections, not fragments. If two top-N chunks share the same section, they are deduplicated — the section is fetched once.

---

### 8. Query Optimisation

**File:** `src/phflow/generation/prompts.py`, `src/phflow/generation/chain.py`

**Retrieval query**

The user's raw query is optionally rewritten (see §6.3) before the retrieval encode step. Only the (potentially rewritten) retrieval query is sent to FAISS; the original question is used for prompt assembly.

**Adaptive history window**

The prompt includes conversation history to resolve follow-up references. The window size adapts to the query:

- **Default (`HISTORY_WINDOW = 0`):** no prior turns included (eliminates hallucination risk on fresh queries)
- **Expanded (2 turns / 4 messages):** automatically activated when the query contains a back-reference pronoun or phrase — `"that"`, `"it"`, `"its"`, `"same"`, `"the section"`, `"the requirement"`, `"previous"`, etc.

**Retrieval query isolation**

Prior conversation turns are never concatenated into the retrieval query. Early testing showed that mixing prior turns into the retrieval query degraded precision by pulling the query vector toward the previous topic.

---

### 9. Prompt Assembly

**File:** `src/phflow/generation/prompts.py`, `src/phflow/generation/citations.py`

The system prompt is loaded from `prompts/system_prompt.txt` at runtime (falls back to an inline default if the file is missing).

The assembled prompt has four sections in order:

```
INSTRUCTIONS:
{system_prompt}

CONVERSATION HISTORY:          ← adaptive: 0 turns normally, 2 turns on back-references
User: {prior question}
Assistant: {prior answer}

CONTEXT:
[SOURCE 1] 21 CFR § 211.182 — § 211.182 Equipment cleaning and use log.
Part 211: ... | Subpart J: ...
Last amended: 2016-11-29
————————————————————————————————————————————————————————————
{full section text}

[SOURCE 2] ...

QUESTION:
{user's question}

Answer:
```

**System prompt rules**

The system prompt (`prompts/system_prompt.txt`) enforces eight rules:

| Rule | Behaviour |
|---|---|
| Context-only | Every claim must come from the CONTEXT block. No training knowledge. |
| Insufficient context | If CONTEXT has no relevant information, respond with exactly the standard insufficient-context sentence and nothing else. |
| No cross-inference | Do not combine or infer definitions across sentences — if a term isn't explicitly defined, refuse to answer. |
| Partial context | If CONTEXT partially addresses the question, answer from what is available and explicitly state what is missing. |
| Operational language | Map warehouse/operational terms to CFR equivalents (pallet→container, leakage→broken seal, etc.). |
| Answer format | Formal prose paragraphs. No bullets. No CFR section numbers or `[SOURCE N]` labels in the answer body. |
| Source declaration | Every response must end with `SOURCES_USED: 1, 3` (1-based source indices). This is mandatory. |
| Output only | No reasoning, no self-corrections, no deliberation, no repetition. |

**Context block format**

Each source in the CONTEXT block is labelled `[SOURCE N]` and includes:
- Section number and title
- Part and Subpart location
- Last amendment date
- Full section text (from parent expansion)

Labels are 1-based. The `SOURCES_USED:` declaration at the end of the answer maps these indices back to the actual sections, allowing the API to return only the sources the model actually used.

---

### 10. Generation and Output Cleaning

**File:** `src/phflow/generation/llm.py`, `src/phflow/generation/chain.py`

#### 10.1 LLM Client (`llm.py`)

The `LlamaCppClient` posts to the llama.cpp `/completion` endpoint:

```json
{
  "prompt":            "...",
  "n_predict":         768,
  "temperature":       0.1,
  "top_p":             0.9,
  "n_ctx":             65536,
  "stream":            false,
  "stop":              ["<|im_end|>", "</s>", "<|eot_id|>", ...],
  "repeat_penalty":    1.1,
  "repeat_last_n":     256,
  "frequency_penalty": 0.1
}
```

**Temperature `0.1`** — near-deterministic. Regulatory answers must be factual, not creative.

**Repeat penalty** — `repeat_penalty=1.1` with a 256-token look-back window, combined with `frequency_penalty=0.1`, penalises the model for repeating the same token sequences. This prevents two observed failure modes:
1. The `SOURCES_USED:` line looping indefinitely
2. The model echoing the question + context back as its answer body

**Stop sequences** — EOS and role tokens across major model families are included (`<|im_end|>`, `</s>`, `<|endoftext|>`, `<|eot_id|>`, etc.) plus prompt section labels (`INSTRUCTIONS:`, `CONTEXT:`, `QUESTION:`) to stop generation cleanly.

#### 10.2 Output Cleaning (`_clean`)

After generation, `_clean()` applies multiple passes:

1. **`<think>` block removal** — strips reasoning blocks emitted by chain-of-thought models (Qwen3, DeepSeek-R1)
2. **EOS token stripping** — leaked `<|im_end|>`, `</s>`, etc. are stripped from start and end
3. **Section-label stripping** — prompt section headers (`INSTRUCTIONS:`, `CONTEXT:`, `QUESTION:`, `Answer:`) echoed by the model are stripped

#### 10.3 SOURCES_USED Parsing

`_parse_used_sources()` in `chain.py` processes the cleaned answer:

- Scans **all lines** for `SOURCES_USED:` and takes the **last** occurrence. The last is preferred because without repeat_penalty the model sometimes generates two complete answers; the second is typically better.
- Matching is **case-insensitive** and accepts both `SOURCES_USED:` and `SOURCES USED:` (space variant emitted by some models)
- Parses the 1-based source indices (e.g. `"1, 3"`) and maps them to the actual `source_summary` entries
- Strips the `SOURCES_USED:` line and everything after it from the answer
- Strips any residual `[SOURCE N]` or `Source N` markers from the answer body

---

### 11. Answer Integrity Guards

**File:** `src/phflow/generation/chain.py`

After parsing, the answer passes through several guards. Any failure replaces the answer with the standard insufficient-context message and clears sources:

| Guard | Condition | Failure mode caught |
|---|---|---|
| **No sections** | `not sections` after retrieval | Retrieval failure → LLM never called, `_INSUFFICIENT` returned immediately |
| **Empty answer** | `not answer` | Model returned nothing after cleaning |
| **Bare CFR header** | Answer matches a section title/number pattern | Model echoed a context block label as its entire answer |
| **Too short** | `len(answer) < 60` | Parametric one-liner (`"2–8°C"`, `"✘ No."`) from training memory |
| **Prompt echo** | `"CONTEXT:"` / `"QUESTION:"` / `"INSTRUCTIONS:"` in answer | Model reproduced prompt structure in its answer body |
| **Template placeholder** | `"[fill in"` / `"[insert"` / `"[answer here"` in answer | Model produced a fill-in-the-blank stub instead of a real answer |
| **Reasoning leak** | `"<think>"` in answer | Chain-of-thought reasoning leaked into the output body |

The **no-sections short-circuit** is critical: when all retrieved chunks are filtered out, the LLM must not be called. Without this guard, the LLM receives `"No relevant regulatory sections found."` as its context and hallucinates a confident answer from training memory.

---

### 12. Session Management

**File:** `src/phflow/api/session_store.py`

Sessions persist the conversation history (`user` + `assistant` message pairs) across turns. Two backends are available:

| Backend | Activation | Behaviour |
|---|---|---|
| `InMemorySessionStore` | Default (no config needed) | Module-level dict; sessions lost on restart |
| `RedisSessionStore` | Set `REDIS_URL=redis://localhost:6379` | JSON-serialized in Redis with 24-hour TTL; sessions survive restarts |

The API selects the backend automatically at startup. If `REDIS_URL` is set but Redis is unreachable, it falls back to in-memory silently.

Both backends trim history to `SESSION_MAX_MESSAGES` (default: 100) to prevent unbounded memory growth.

**Streaming and session history**

Both streaming and non-streaming responses save the full conversation turn to the session store after completion. Streaming responses include an `X-Session-Id` response header so clients can continue the session.

---

### 13. Per-Query Debug Logging

**File:** `src/phflow/utils/query_logger.py`

Every query writes one timestamped JSON file to `query_logs/`:

```
query_logs/
└── 20260309_141457_691117__What_are_the_GMP_requirements_for.json
```

Each file captures (in order):
1. Original query
2. Retrieval query (after rewrite, if enabled)
3. Auto-extracted filters (if any were detected)
4. Applied scope (if default scope was used)
5. Retrieval candidates: section number, title, RRF score (top-N)
6. Sections passed to LLM: section number, title, RRF score, full_text char count
7. Assembled prompt — stored as an **array of lines** for readability
8. Raw LLM output — stored as an **array of lines** for readability
9. Final answer — stored as an **array of lines** for readability
10. Sources declared by LLM

Structured JSON enables aggregate analysis (e.g. "what % of queries had 0 sections after retrieval?") and is readable in any JSON viewer or text editor.

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
| `LLAMA_CPP_URL` | `http://localhost:8080` | llama.cpp server URL |
| `LLAMA_CPP_TEMPERATURE` | `0.1` | LLM temperature (low = factual) |
| `LLAMA_CPP_NUM_CTX` | `65536` | Context window — must match `-c` on llama-server |
| `LLAMA_CPP_MAX_TOKENS` | `768` | Max response tokens |
| `INITIAL_RETRIEVAL_K` | `10` | Candidates after RRF fusion |
| `RETRIEVAL_TOP_N` | `5` | Sections passed to LLM |
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

## License

This project is for research and educational use. The regulatory data (ECFR Title 21) is published by the U.S. Government Printing Office and is in the public domain.
