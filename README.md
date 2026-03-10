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
│  Query ──► BGE embed (w/ prefix) ──► Dense vector ────┐         │
│        ──► BM25 (fastembed)      ──► Sparse vector ───┤  RRF    │
│                       (text + table chunks)            └►Qdrant  │
│                                     BGE-Reranker-base ──►top-5  │
│                                     Parent expansion ──► LLM    │
└──────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│   Qdrant  (port 6333)           │
│   Collection: ecfr_title21      │
│   ~50,000 structural chunks     │
│   Dense 768-dim + BM25 sparse   │
└─────────────────────────────────┘
```

### Tech Stack

| Component | Technology |
|---|---|
| UI | Vanilla HTML/CSS/JS (served by FastAPI) |
| Embedding | `BAAI/bge-base-en-v1.5` (768-dim, asymmetric query prefix) |
| Sparse search | `fastembed` — `Qdrant/bm25` |
| Vector DB | Qdrant (hybrid dense + sparse, HNSW, on-disk BM25) |
| Reranker | `BAAI/bge-reranker-base` (cross-encoder, sigmoid scoring) |
| LLM | llama.cpp server (any GGUF model) |
| API | FastAPI + uvicorn |

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
│   │   ├── chunker.py            # structural chunker (one chunk per top-level CFR paragraph)
│   │   ├── metadata.py           # versions.json enrichment
│   │   └── pipeline.py           # ingestion orchestrator
│   ├── embeddings/encoder.py     # BGE sentence-transformer wrapper (asymmetric encode_query)
│   ├── vectorstore/
│   │   ├── client.py             # Qdrant client + upsert helpers
│   │   └── schema.py             # collection + payload index setup
│   ├── retrieval/
│   │   ├── hybrid.py             # dense + sparse RRF → rerank → parent expand
│   │   ├── reranker.py           # BGE-Reranker-base (cross-encoder, sigmoid scoring)
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

The full concatenation of all paragraphs in the section is stored in the Qdrant payload as `parent_text`. It is never embedded. It exists so that, after retrieval identifies the relevant section via its children, the full section text (not just the matching chunk) is fetched and passed to the LLM.

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

Both query and passage vectors are L2-normalised before storage, so cosine similarity reduces to dot product, enabling fast HNSW search.

---

### 5. Indexing

**File:** `src/phflow/vectorstore/schema.py`

**Vector index — Dense (HNSW)**

```
Distance:       Cosine
Dimensions:     768
HNSW m:         16       (number of edges per node — controls graph connectivity)
HNSW ef_construct: 200  (beam width during index build — higher = better recall at build cost)
```

`m=16` is the standard production setting for HNSW. `ef_construct=200` is 2× the default, chosen because the regulatory corpus has many near-duplicate sections (e.g. Parts 211, 213, 225 share nearly identical text) and higher ef improves graph quality in dense similarity clusters.

**Vector index — Sparse (BM25)**

BM25 sparse vectors are stored on-disk (`on_disk=True`) to avoid memory pressure. The `fastembed` library tokenises chunks using the `Qdrant/bm25` model and encodes them as sparse `{token_id: tf-idf weight}` vectors. These complement dense retrieval by matching on exact regulatory terminology (section numbers, defined terms, chemical names).

**Payload fields and indexes**

Every chunk stores a rich metadata payload. The following fields are indexed for `O(log n)` filtered retrieval:

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

#### 6.1 Hybrid Search (Dense + BM25 via RRF)

Both retrieval signals are computed in parallel and fused natively inside Qdrant using Reciprocal Rank Fusion (RRF):

```
Query
  │
  ├─► encode_query(prefix + query)  →  dense vector
  │                                         │
  └─► bm25.query_embed(query)        →  sparse vector
                                            │
              Qdrant query_points (FusionQuery / RRF)
              ┌────────────────────────────────────────┐
              │  Prefetch: dense top-80   score desc   │
              │  Prefetch: sparse top-80  score desc   │
              │                                        │
              │  RRF merge  →  top-40 candidates      │
              └────────────────────────────────────────┘
```

**Why RRF instead of score interpolation?**

Dense and BM25 scores exist on incompatible scales. RRF avoids the need to tune a weighting factor `α` by working purely on rank positions:

```
RRF(d) = Σ  1 / (k + rank_i(d))
```

with `k=60` (Qdrant default). A chunk ranked 1st in both lists gets a higher RRF score than one ranked 5th in both, regardless of the raw score values. This is robust to scale differences and avoids expensive calibration.

**Authority chunk exclusion**

A permanent filter is AND-ed into every search that excludes `chunk_type="authority"` chunks. These statutory citation blocks score high on BM25 for almost any regulatory query (they contain part numbers, U.S.C. citations) but carry no substantive content.

**Prefetch limits**

Each prefetch (dense and sparse) fetches `initial_k × 2 = 80` candidates. After RRF merge, the top `initial_k = 40` are passed to the reranker. The 2× prefetch factor ensures that both retrieval signals contribute meaningfully to the final candidate set — a chunk that scores 41st in dense but 3rd in BM25 would be lost if prefetch matched `initial_k` exactly.

#### 6.2 Metadata Filtering

Any combination of the following filters can be applied pre-search (pushed into the Qdrant `filter` parameter, not post-filtered):

| Filter | Field | Example |
|---|---|---|
| CFR Part | `part_number` (int) | `part_number=211` |
| Chapter | `chapter_number` (keyword) | `chapter_number="I"` |
| Subpart | `subpart_letter` (keyword) | `subpart_letter="B"` |
| Chunk type | `chunk_type` (keyword, `MatchAny`) | `chunk_types=["text","table"]` |
| Amendment date | `latest_amendment_date` (range) | `amended_after="2020-01-01"` |
| Has table | `has_table` (bool) | `has_table=True` |
| Exclude repealed | `is_removed` (bool) | always `is_removed=False` |

Filters are AND-composed. The `is_removed=False` filter is always active to exclude repealed sections.

---

### 7. Reranking

**File:** `src/phflow/retrieval/reranker.py`

**Model:** `BAAI/bge-reranker-base`

A cross-encoder is fundamentally different from a bi-encoder (embedding model). Instead of encoding the query and passage independently and comparing vectors, the cross-encoder sees both together:

```
[CLS] query [SEP] passage [SEP]  →  single relevance logit
```

This joint attention over both texts enables much more precise relevance judgement at the cost of not being indexable — it must score every (query, candidate) pair individually.

**Scoring**

```python
logits = model(**inputs).logits          # shape [batch, 1]
scores = torch.sigmoid(logits[:, 0])     # → [0, 1] relevance probability
```

`sigmoid(logit)` converts the raw classification output to a calibrated probability in [0, 1]. A score of 0.5 means the model is uncertain; scores above ~0.6 indicate genuine relevance.

**Threshold filtering**

Only candidates with `score >= reranker_score_threshold` (default `0.25`) survive. This threshold is intentionally low because:

1. The reranker is a general-purpose model, not fine-tuned on regulatory text, so calibrated scores for regulatory content cluster in the 0.3–0.6 range even for genuinely relevant passages
2. A false negative (relevant section dropped) produces an empty context → `_INSUFFICIENT` answer, which is worse than passing a slightly weaker section to the LLM

Top `reranker_top_n` (default 5) survivors are returned, sorted by score descending.

**Batch inference**

All 40 candidates are scored in a single batched call with padding and truncation to `max_length=512`. This is ~8× faster than scoring one pair at a time on CPU.

---

### 8. Parent Expansion

**File:** `src/phflow/retrieval/hybrid.py` → `scroll_by_node`

After reranking, the top-5 results are individual *chunks* (one paragraph group each). The LLM needs the *full section* for context — a single `(b)` sub-paragraph without its surrounding `(a)` and `(c)` may be ambiguous or incomplete.

For each unique `section_node` in the reranked results:

1. Scroll all chunks sharing that `section_node` from Qdrant
2. Concatenate their `chunk_text` in `chunk_index` order
3. Replace the single matched chunk with the full section text

This ensures the LLM always receives complete regulatory sections, not fragments. If two reranked chunks share the same section, they are deduplicated — the section is fetched once.

---

### 9. Query Optimisation

**File:** `src/phflow/generation/prompts.py`, `src/phflow/generation/chain.py`

**Current state: pass-through**

The user's raw query is passed directly to both the retrieval encoder and the reranker. No rewriting or expansion is performed before retrieval.

This preserves latency (one LLM call instead of two) but means that conversational workplace language (`"fridge"`, `"sign off on a digital form"`, `"trusted supplier"`) has a vocabulary gap from formal CFR regulatory text. The vocabulary gap is the primary cause of retrieval misses for colloquial queries.

**History window**

Only the most recent conversation turn (user + assistant pair) is prepended to the generation prompt for follow-up context. Full history is not prepended to the retrieval query — early testing showed that concatenating prior turns into the retrieval query degraded precision by pulling the query vector toward the previous topic rather than the current question.

**Future: LLM-based retrieval rewrite**

The codebase includes `build_retrieval_rewrite_prompt()` in `prompts.py` — a prompt designed to convert colloquial queries to formal CFR terminology before retrieval, using a short LLM call with `max_tokens=80`. This was implemented but deferred due to the latency cost of an additional inference round-trip before retrieval.

---

### 10. Prompt Assembly

**File:** `src/phflow/generation/prompts.py`, `src/phflow/generation/citations.py`

The assembled prompt has four sections in order:

```
INSTRUCTIONS:
{system_prompt}

CONVERSATION HISTORY:          ← optional, last 1 turn only
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

The system prompt (`SYSTEM_PROMPT` in `prompts.py`) enforces six rules:

| Rule | Behaviour |
|---|---|
| Context-only | Every claim must come from the CONTEXT block. No training knowledge. |
| Insufficient context | If CONTEXT has no relevant information, respond with exactly the standard insufficient-context sentence and nothing else. |
| Partial context | If CONTEXT partially addresses the question, answer from what is available and explicitly state what is missing. |
| Lengthy context | Distil long contexts — do not reproduce large blocks verbatim. |
| Answer format | Formal prose paragraphs. No bullets. No CFR section numbers or `[SOURCE N]` labels in the answer body. |
| Source declaration | Every response must end with `SOURCES_USED: 1, 3` (1-based source indices). This is mandatory. |

**Context block format**

Each source in the CONTEXT block is labelled `[SOURCE N]` and includes:
- Section number and title
- Part and Subpart location
- Last amendment date
- Full section text (from parent expansion)

Labels are 1-based. The `SOURCES_USED:` declaration at the end of the answer maps these indices back to the actual sections, allowing the API to return only the sources the model actually used.

**Conversation history window**

`HISTORY_WINDOW = 1` — only the last user+assistant pair is included. This gives the model enough context to resolve pronouns and follow-up references (`"What about that section?"`) without polluting the prompt with older conversation turns that could bias retrieval or confuse the model.

---

### 11. Generation and Output Cleaning

**File:** `src/phflow/generation/llm.py`, `src/phflow/generation/chain.py`

#### 11.1 LLM Client (`llm.py`)

The `LlamaCppClient` posts to the llama.cpp `/completion` endpoint:

```json
{
  "prompt":            "...",
  "n_predict":         2048,
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

**Temperature `0.1`** — near-deterministic. Regulatory answers must be factual, not creative. Low temperature eliminates variance in phrasings of the same factual claim.

**Repeat penalty** — `repeat_penalty=1.1` with a 256-token look-back window, combined with `frequency_penalty=0.1`, penalises the model for repeating the same token sequences. This prevents two observed failure modes:
1. The `SOURCES_USED:` line looping indefinitely (model repeats `"211.42, 211.80"` hundreds of times)
2. The model echoing the question + context back as its answer body

**Stop sequences** — EOS and role tokens across major model families are included (`<|im_end|>`, `</s>`, `<|endoftext|>`, `<|eot_id|>`, etc.) to stop generation cleanly regardless of which model is loaded.

#### 11.2 Output Cleaning (`_clean`)

After generation, `_clean()` applies three passes in order:

1. **`<think>` block removal** — Regex `re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)` strips reasoning blocks emitted by chain-of-thought models (Qwen3, DeepSeek-R1). These blocks may be empty (`<think>\n</think>`) or contain multi-line reasoning. All occurrences are removed, not just the first.

2. **EOS token stripping** — Leaked `<|im_end|>`, `</s>`, etc. are stripped from the start and end of the output in a loop until none remain.

3. **Section-label stripping** — Prompt section headers (`INSTRUCTIONS:`, `CONTEXT:`, `QUESTION:`, `Answer:`) that the model echoed are stripped from start and end.

#### 11.3 SOURCES_USED Parsing

`_parse_used_sources()` in `chain.py` processes the cleaned answer:

- Scans **top-down** for the **first** `SOURCES_USED:` line. Taking the first occurrence is correct because the model should emit this line once at the end of its answer — any subsequent occurrences are part of a repetition loop and are discarded along with everything after them.
- Parses the 1-based source indices (e.g. `"1, 3"`) and maps them to the actual `source_summary` entries
- Strips the `SOURCES_USED:` line and everything after it from the answer
- Strips any residual `[SOURCE N]` markers from the answer body

---

### 12. Answer Integrity Guards

**File:** `src/phflow/generation/chain.py`

After parsing, the answer passes through five guards. Any failure replaces the answer with the standard insufficient-context message and clears sources:

| Guard | Condition | Failure mode caught |
|---|---|---|
| **No sections** | `not sections` after reranking | Retrieval failure → LLM never called, `_INSUFFICIENT` returned immediately |
| **Empty answer** | `not answer` | Model returned nothing after cleaning |
| **Bare CFR header** | Matches `^21\s+CFR\s+§?\s*[\d.]+…$` | Model echoed a context block label as its entire answer |
| **Too short** | `len(answer) < 60` | Parametric one-liner (`"2–8°C"`, `"✘ No."`) from training memory |
| **Prompt echo** | `"CONTEXT:" in answer` or `"QUESTION:" in answer` | Model reproduced prompt structure in its answer body (catastrophic loop) |
| **Template placeholder** | `"[fill in"` / `"[insert"` / `"[answer here"` in answer | Model produced a fill-in-the-blank stub instead of a real answer |

The **no-sections short-circuit** is critical: when the reranker filters out all candidates (common when the query vocabulary has no overlap with any retrieved section), the LLM must not be called. Without this guard, the LLM receives `"No relevant regulatory sections found."` as its context and hallucinates a confident answer from training memory.

---

### 13. Per-Query Debug Logging

**File:** `src/phflow/utils/query_logger.py`

Every query writes a timestamped log to `query_logs/` capturing the full pipeline state:

```
query_logs/
└── 20260309_141457_691117__My_coworker_said_we_don_t_need_to_do_a_f.log
```

Each log contains (in order):
1. Original query
2. Retrieval query (after any transformation)
3. Pre-rerank candidates: section number, title, RRF score (top 40)
4. Post-rerank results: section number, title, reranker probability (top 5)
5. Sections passed to LLM: section number, title, reranker score, full_text char count
6. Full assembled prompt
7. Raw LLM output (before any cleaning or guard evaluation)
8. Final answer (after all cleaning and guards)
9. Sources declared by LLM

These logs are the primary debugging tool. Reading a log file immediately reveals whether a failure is a retrieval miss (wrong sections in pre-rerank), a reranker failure (correct sections dropped below threshold), a prompt issue (assembled prompt malformed), or an LLM output issue (degenerate answer).

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

## Configuration

All settings are in `.env` (see `.env.example`):

| Variable | Default | Description |
|---|---|---|
| `EMBEDDING_MODEL` | `BAAI/bge-base-en-v1.5` | Sentence-transformer model |
| `EMBEDDING_DEVICE` | `cuda` | `cpu`, `cuda`, or `mps` |
| `QDRANT_URL` | `http://localhost:6333` | Qdrant server URL |
| `COLLECTION_NAME` | `ecfr_title21` | Qdrant collection |
| `LLAMA_CPP_URL` | `http://localhost:8080` | llama.cpp server URL |
| `LLAMA_CPP_TEMPERATURE` | `0.1` | LLM temperature (low = factual) |
| `LLAMA_CPP_NUM_CTX` | `65536` | Context window — must match `-c` on llama-server |
| `LLAMA_CPP_MAX_TOKENS` | `2048` | Max response tokens |
| `RERANKER_MODEL` | `BAAI/bge-reranker-base` | Cross-encoder reranker |
| `INITIAL_RETRIEVAL_K` | `40` | Candidates before reranking |
| `RERANKER_TOP_N` | `5` | Sections passed to LLM |
| `RERANKER_SCORE_THRESHOLD` | `0.25` | Min reranker yes-probability (filters weak matches) |
| `CHUNK_MAX_TOKENS` | `512` | Overflow guard for structural chunking |

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
