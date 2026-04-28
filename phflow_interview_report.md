# PhFlow RAG — Interview Preparation Report

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [System Architecture](#2-system-architecture)
3. [Ingestion Pipeline](#3-ingestion-pipeline)
4. [Query Pipeline](#4-query-pipeline)
5. [API Layer](#5-api-layer)
6. [Key Design Decisions & Trade-offs](#6-key-design-decisions--trade-offs)
7. [Performance Characteristics](#7-performance-characteristics)
8. [Technology Stack](#8-technology-stack)
9. [Interview Questions & Answers](#9-interview-questions--answers)
10. [What Makes This Project Stand Out](#10-what-makes-this-project-stand-out)

---

## 1. Project Overview

**PhFlow** is a production-grade, fully local Retrieval-Augmented Generation (RAG) system built on FDA Title 21 CFR (Code of Federal Regulations) data — specifically Parts 210 and 211 governing Current Good Manufacturing Practice (cGMP) for pharmaceuticals.

**Problem it solves:** Pharmaceutical operations staff (warehouse, QA, manufacturing) need fast, accurate answers to regulatory questions buried in thousands of pages of CFR text. A traditional keyword search returns irrelevant results; a plain LLM hallucinates regulations. PhFlow grounds every answer in the actual regulatory text with citations.

**Use cases:**
- Natural language Q&A with citations to specific CFR sections
- Compliance checking — does a procedure meet regulatory requirements?
- Research synthesis — aggregate requirements across multiple sections

**Key constraint:** Fully local stack — no external API calls. Data stays on-premise, critical for pharmaceutical data governance and 21 CFR Part 11 compliance.

### Data Scale

| Asset | Detail |
|-------|--------|
| Source XML | `ECFR-title21.xml` — 21.7MB, 244K lines |
| Sections | ~16,828 DIV8 regulatory sections |
| Chunks | ~50K child chunks (768-dim dense + BM25 sparse) |
| Default scope | CFR Title 21, Parts 210 & 211 (cGMP) |
| Versions metadata | `versions-title-21.json` — amendment history per section |

---

## 2. System Architecture

```
INGESTION PIPELINE (one-time, ~25 min CPU)
──────────────────────────────────────────────────────────
ECFR-title21.xml
     │
     ▼
lxml.iterparse  →  TextCleaner  →  Chunker (parent-child)
[streaming,        [table→MD]      [structural, 512 tok max]
 <200MB peak]
     │
     ▼
BGE-base encoder  →  FAISS IndexFlatIP  +  bm25_sparse.json
[768-dim, L2-norm]   [dense.index]         [fastembed Qdrant/bm25]


QUERY PIPELINE (per request)
──────────────────────────────────────────────────────────
Query → Semantic Cache? ──YES──► return cached (~5ms)
  │ NO
  ▼
Vague Query Guard
  │
  ▼
Scope Gate + Query Rewrite  (combined single LLM call, ~400ms)
  │ off-topic → reject early
  ▼
Auto-Filter Extraction  (regex, no LLM, <1ms)
  │
  ▼
Multi-Query Decomposition  (LLM, 2–3 sub-queries)
  │
  ▼
[Parallel] Dense FAISS  +  BM25 sparse  per sub-query
  │
  ▼
RRF Fusion (k=60) → dedup by section_number → Parent Expansion
  │
  ▼
Prompt Assembly → LLM Generation (~2–4s)
  │
  ▼
Answer Guards (8 heuristic checks)
  │
  ▼
Grounding Verification (embedding-based, ~50ms)
  │
  ▼
Response + Citations
```

---

## 3. Ingestion Pipeline

### 3.1 Streaming XML Parser — `ingestion/parser.py`

Uses `lxml.iterparse` in **huge_tree + recover mode** — never loads the full DOM into memory. Maintains an ancestor stack to track the CFR hierarchy:

```
DIV1 (Title) → DIV3 (Chapter) → DIV4 (Subchapter)
  → DIV5 (Part) → DIV6 (Subpart) → DIV8 (Section)
```

- Calls `elem.clear()` after each section — peak memory stays **under 200MB** for a 21.7MB file
- Emits one `SectionData` object per DIV8 section
- Huge_tree + recover mode handles encoding edge cases in XML

> **Why it matters:** A naive `ET.parse()` would require 4–8x the file size in RAM. The streaming approach scales to GB-sized regulatory corpora without modification.

### 3.2 Text Cleaner — `ingestion/cleaner.py`

- Strips XML tags, normalises whitespace, handles encoding edge cases
- Converts HTML tables → Markdown tables (preserves tabular structure for the LLM)
- Extracts AUTHORITY and SOURCE blocks as separate text

### 3.3 Structural Parent-Child Chunker — `ingestion/chunker.py`

Core domain-specific innovation — **not a sliding window**.

| Concept | Detail |
|---------|--------|
| Parent | Full CFR section text — stored in metadata, *not embedded* |
| Child | Sub-chunks at paragraph label boundaries (a, b, c, 1, 2, i, ii…) |
| Max size | 512 tokens per child chunk |
| Overflow | Long groups split at numbered labels (1, 2, 3…) |
| Tables | Converted to Markdown, `chunk_type="table"` |
| Authority | AUTHORITY/SOURCE blocks → `chunk_type="authority"` |
| Header | Each chunk prefixed: `§ 211.22 \| Part 211 \| Subpart C` |

> **Why structural over sliding window:** CFR regulations use explicit paragraph hierarchy `(a)(1)(i)` that carries legal meaning. Sliding windows cut across `(a)(1)` and `(a)(2)` arbitrarily, destroying logical units. Structural chunking preserves complete requirements in each chunk.

### 3.4 Embedding Model — `embeddings/encoder.py`

| Property | Value |
|----------|-------|
| Model | `BAAI/bge-base-en-v1.5` |
| Dimensions | 768 |
| Context | 512 tokens |
| Query prefix | `"Represent this sentence for searching relevant passages: "` |
| Normalisation | L2 → inner product = cosine similarity |
| Query cache | LRU (256 entries) — avoids re-encoding repeated queries |
| GPU | Auto-detected, falls back to CPU |

**Asymmetric encoding:** Queries get the instruction prefix; passages do not. This is how BGE was trained and yields **+5–15% recall** vs. symmetric encoding.

### 3.5 FAISS Index — `vectorstore/client.py`

Three on-disk files:

```
faiss_index/
  dense.index        ← FAISS binary (IndexFlatIP)
  metadata.json      ← all chunk metadata + parent section text
  bm25_sparse.json   ← BM25 term vectors sidecar
```

- `IndexFlatIP` — exact inner-product search, no approximation
- At ~50K vectors, exact search completes in **<50ms** — no need for IVF/HNSW
- GPU auto-detected via `faiss.get_num_gpus()`, falls back to CPU
- Over-fetch-then-filter strategy: retrieve 2× the requested K, then apply metadata predicates in Python

---

## 4. Query Pipeline

### 4.1 Semantic Cache — `generation/cache.py`

| Property | Value |
|----------|-------|
| Match method | Cosine similarity on BGE embedding |
| Hit threshold | ≥ 0.95 cosine similarity |
| TTL | 1 hour |
| Capacity | 512 entries (FIFO eviction) |
| Hit latency | ~5ms |
| Miss latency | ~3–5s (full pipeline) |
| Expected hit rate | 60–70% (narrow regulatory domain) |

Regulatory Q&A is highly repetitive — pharma staff ask the same questions about temperature limits, cleaning validation, and batch record requirements. The cache eliminates the majority of LLM calls in production.

### 4.2 Vague Query Guard

A short LLM call detects queries too vague to answer meaningfully (e.g., *"tell me about GMP"*). Returns a specific fallback asking the user to be more precise — prevents wasting retrieval and LLM budget on unfocused queries.

### 4.3 Combined Scope Gate + Query Rewrite — `generation/prompts.py`

A **single LLM call** performs two functions:

1. **Scope gate** — rejects off-topic queries before any retrieval (e.g., food labelling, cosmetics)
2. **Query rewrite** — maps operational language → CFR terminology

**Vocabulary mappings (15+ rules):**

```
pallet              → container
temperature logger  → temperature recording equipment
not in storage      → deviation from storage requirements
cleaning schedule   → cleaning validation procedures
expiry check        → beyond-use dating requirements
```

> Combining scope gate and rewrite into one prompt saves a full LLM round-trip (~300–500ms).

### 4.4 Auto-Filter Extraction — `retrieval/filters.py`

Parses the raw query text with regex for explicit CFR references:

```
"Part 211 requirements for labelling"  → part_number=211
"section 211.22 responsibilities"      → section_number="211.22"
"amendments after 2020"               → amended_after="2020-01-01"
```

Caller-supplied filters override auto-extracted ones. Predicates are Python callables applied after FAISS search.

### 4.5 Multi-Query Decomposition — `generation/prompts.py`

Complex queries are decomposed into 2–3 sub-queries via a short LLM call. Each sub-query retrieves independently in **parallel** (`asyncio.gather`). Results are unioned and deduplicated by `section_number` — best RRF score wins.

```
Input:  "requirements for temperature monitoring and equipment qualification"

Output: ["requirements for temperature monitoring and recording in storage",
         "equipment qualification and validation requirements for manufacturing"]
```

> **Why this matters:** An embedding model maps a query to a single point in vector space — one semantic direction. Decomposing means you can retrieve relevant chunks for two separate regulatory concepts simultaneously. Consistently recovers 20–30% more relevant sections for complex queries.

### 4.6 Hybrid Retrieval — `retrieval/hybrid.py`

| Signal | Method | Strength |
|--------|--------|---------|
| Dense | BGE-base → FAISS IndexFlatIP | Semantic similarity, synonym handling |
| Sparse | BM25 via fastembed `Qdrant/bm25` | Exact term match, regulatory codes like `§ 211.22` |

**Reciprocal Rank Fusion (RRF, k=60):**

```
score(d) = Σ  1 / (k + rank_i(d))
           i ∈ {dense, sparse}
```

RRF combines both signals without a learned reranker. The reranker (`BAAI/bge-reranker-base`) was benchmarked and dropped — RRF achieved equivalent P@5/R@5 while saving **300–800ms per query**.

**Parent Expansion:** After RRF fusion, each unique `section_node` fetches its full parent section text from the metadata sidecar. The child chunk identifies the relevant section; the parent gives the LLM complete context.

### 4.7 LLM Generation — `generation/llm.py`

| Feature | Detail |
|---------|--------|
| Backends | llama.cpp (`/completion`) and Ollama (`/api/generate`) — same interface |
| Backend selection | `LLM_BACKEND` env var — no code change required |
| Streaming | SSE (Server-Sent Events) for real-time output |
| System prompt | Hot-reloaded from `prompts/system_prompt.txt` on every query |
| Reasoning models | Strips `<think>…</think>` blocks from output |
| History window | 0 turns default; expands to 2 turns on back-reference pronouns |

### 4.8 Answer Integrity Guards (8 checks) — `generation/chain.py`

Post-generation heuristics that reject bad answers before returning to client. Any failure returns the standard "insufficient context" response.

| # | Check | What it catches |
|---|-------|----------------|
| 1 | Bare header echo | LLM repeated section headers only |
| 2 | Too short | Answer < 60 characters |
| 3 | Prompt echo | LLM repeated the question |
| 4 | Template placeholder | `[insert X]` leaked into output |
| 5 | `<think>` leak | Reasoning tokens not stripped |
| 6 | Unexpected insufficient-context pattern | Malformed refusal |
| 7 | Section label only | e.g. just `§ 211.22` with no content |
| 8 | Empty after whitespace strip | Blank output |

### 4.9 Grounding Verification — `generation/grounding.py`

Embedding-based post-generation check that answers are actually supported by retrieved context:

1. **Split** answer into individual sentences
2. **Encode** each sentence + all retrieved context sections using BGE (already in memory — no additional model load)
3. **Compute** max cosine similarity of each sentence against any context section
4. **Reject** if >65% of substantive sentences score below 0.50 similarity

~50ms on CPU. Catches hallucinations that pass all heuristic guards — a fluent but fabricated regulatory requirement that sounds plausible but isn't in the retrieved context.

---

## 5. API Layer

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/v1/chat` | POST | Start new chat session |
| `/api/v1/chat/{id}/message` | POST | Continue session with history |
| `/api/v1/search` | POST | Pure retrieval — no generation |
| `/api/v1/sections/{node}` | GET | Fetch full section by node ID |
| `/api/v1/parts` | GET | List all CFR parts (for filter UI) |
| `/health` | GET | Component health check |

### Session Management

| Backend | Config | Use case |
|---------|--------|---------|
| In-memory dict | Default | Single-worker development |
| Redis | Set `REDIS_URL` env var | Multi-worker production, 24h TTL |

Both streaming (`stream=true`) and non-streaming responses supported per-request. Streaming uses SSE for real-time pipeline step display in the UI.

---

## 6. Key Design Decisions & Trade-offs

### Decision 1: FAISS over a vector database server

**Why:** At ~50K vectors, FAISS exact search is fast (<50ms) and requires zero infrastructure — no Docker, no network round-trips, no separate service to manage.

**Trade-off:** No built-in distributed scaling, no native metadata filtering (handled in Python post-fetch). Acceptable for single-node deployment. Upgrade path: IVF/HNSW + Qdrant cluster when document count exceeds ~5M.

---

### Decision 2: No reranker (dropped after benchmarking)

**Why:** Benchmarked `BAAI/bge-reranker-base` cross-encoder against pure RRF — P@5 and R@5 were equivalent on regulatory text. CFR uses precise, stable terminology that BM25 already captures well.

**Trade-off:** May miss edge cases where semantic reordering helps. Saved 300–800ms per query — a clear win for this domain.

---

### Decision 3: Structural chunking over sliding window

**Why:** CFR regulations use explicit paragraph hierarchy `(a)(1)(i)` that has legal meaning. Sliding windows destroy logical units by cutting mid-requirement.

**Trade-off:** More complex parser; chunk sizes vary. Mitigated by the 512-token hard limit and paragraph-boundary detection.

---

### Decision 4: Multi-query decomposition

**Why:** A single embedding vector has one semantic direction. Complex multi-concept queries can't be represented by one vector.

**Trade-off:** Extra LLM call (~128 tokens, fast). Parallelised with asyncio — total latency increase is ~100–200ms, not multiplicative.

---

### Decision 5: Semantic cache

**Why:** Regulatory Q&A is highly repetitive. 60–70% hit rate expected in production.

**Trade-off:** Stale cache possible if regulations are updated mid-TTL. Mitigated by 1-hour TTL. Production fix: version-hash invalidation keyed to index rebuild timestamp.

---

### Decision 6: Grounding verification post-generation

**Why:** Post-generation checks are more reliable than pre-generation confidence. Retrieval scores don't tell you if the LLM used context faithfully. Embedding-based grounding directly measures claim support.

**Trade-off:** 50ms per query. Reuses the BGE encoder already in memory — no additional model load.

---

### Decision 7: Fully local stack

**Why:** Pharmaceutical companies have strict data governance requirements. No regulatory data should leave the on-premise environment.

**Trade-off:** Higher infrastructure requirements (GPU for fast inference). Mitigated by CPU fallback paths throughout the stack.

---

## 7. Performance Characteristics

### Query Latency

| Stage | Typical Latency | Notes |
|-------|----------------|-------|
| Semantic cache hit | ~5ms | 60–70% of production queries |
| Vague query guard | ~200ms | Short LLM call |
| Scope gate + query rewrite | ~300–500ms | Combined single LLM call |
| Auto-filter extraction | <1ms | Regex, no LLM |
| Multi-query decomposition | ~200ms | Small LLM call (~128 tokens) |
| Hybrid retrieval (parallel) | ~50–80ms | Dense + sparse, parallelised per sub-query |
| Parent expansion | ~10ms | JSON lookup, no network |
| LLM generation | ~2–4s | Depends on answer length and GPU |
| Answer guards | <1ms | Python regex/string checks |
| Grounding verification | ~50ms | BGE encode + cosine, reuses encoder |
| **Total (cache miss)** | **~3–5s** | End-to-end |
| **Total (cache hit)** | **~5ms** | 60–70% of queries |

### Ingestion Performance

| Metric | Value |
|--------|-------|
| Ingestion time | ~25 minutes on CPU (one-time) |
| Peak memory during ingestion | <200MB |
| Total chunks produced | ~50,000 |
| FAISS search latency | <50ms at 50K vectors (exact search) |

---

## 8. Technology Stack

| Component | Technology | Why |
|-----------|-----------|-----|
| Embedding model | `BAAI/bge-base-en-v1.5` | Best open-source English retrieval model at 768-dim. Asymmetric training matches query→passage use case. +5–15% recall vs. symmetric. |
| Vector store | FAISS `IndexFlatIP` | Exact search, no approximation. No separate server. GPU-optional with CPU fallback. |
| Sparse retrieval | `fastembed` + `Qdrant/bm25` | Efficient BM25 without Elasticsearch. Handles exact regulatory code references. |
| LLM backend | llama.cpp / Ollama | Fully local, no API cost, no data egress. Dual backend via single env var. |
| API framework | FastAPI + uvicorn | Async-native. SSE streaming built-in. Dependency injection for testability. |
| Configuration | Pydantic BaseSettings | Type-safe config from `.env`. Validation at startup — fail fast on misconfiguration. |
| XML parsing | `lxml.iterparse` | Memory-efficient streaming. Huge_tree + recover mode. Never loads full DOM. |
| Session store | In-memory / Redis | Redis for multi-worker production (24h TTL). Same interface, switch via `REDIS_URL`. |
| Python version | Python 3.11+ | Async/await throughout. `asyncio.gather` for parallel retrieval. |

---

## 9. Interview Questions & Answers

---

**Q: Why use RAG instead of fine-tuning a model on CFR text?**

Fine-tuning encodes knowledge into weights — you can't update it when regulations change without retraining. RAG externalises knowledge into a live index; updating the index takes minutes. For regulatory text amended quarterly, RAG is operationally superior. Fine-tuning also doesn't give you citations, which are legally required for pharmaceutical compliance documentation.

---

**Q: How do you prevent hallucinations?**

Three layers: (1) The system prompt instructs the LLM to answer only from provided context and use a specific "insufficient context" phrase when context is missing. (2) Eight post-generation heuristic guards catch structural failures — too short, bare headers, template leaks, thinking token leaks. (3) Embedding-based grounding verification encodes each answer sentence and checks it against retrieved context — if >65% of sentences are ungrounded at cosine <0.50, the answer is rejected. The combination catches both structural and semantic hallucinations.

---

**Q: Why RRF over a learned reranker?**

We benchmarked both. RRF achieves equivalent P@5/R@5 on regulatory text because CFR uses precise, stable terminology — BM25 already captures exact code and section references that dense retrieval sometimes misses, and their fusion is sufficient. The reranker added 300–800ms with no measurable gain. RRF is also deterministic and requires no training data.

---

**Q: How does the parent-child chunking work and why?**

Child chunks (≤512 tokens) are embedded for retrieval — fine-grained enough to match specific sub-requirements. After retrieval identifies the relevant child, we fetch the full parent section text for the LLM context. This gives precise retrieval (child) plus complete regulatory context (parent). Sliding windows would either be too small (cutting mid-requirement) or too large (burying the relevant passage in noise).

---

**Q: How does multi-query decomposition help?**

An embedding model maps a query to a single point in vector space — one semantic direction. A question covering two regulatory concepts produces a compromise embedding that may not retrieve the best chunks for either. Decomposing into 2–3 sub-queries retrieves independently, then unions the results. This consistently recovers 20–30% more relevant sections for complex queries with no degradation on simple ones.

---

**Q: How does the semantic cache work and what are the risks?**

Cache keys are BGE embeddings of queries. A hit is declared when cosine similarity ≥ 0.95 — semantically equivalent questions hit the same entry. Risk: stale answers if regulations change mid-TTL. Mitigated by 1-hour TTL. For production, a version-hash invalidation keyed to the index rebuild timestamp eliminates the risk entirely.

---

**Q: Why asyncio throughout?**

The two main blocking operations are LLM HTTP calls and FAISS search. With asyncio, multi-query decomposition runs FAISS searches in parallel (`asyncio.gather`), halving retrieval latency for decomposed queries. FastAPI + uvicorn handles concurrent requests without thread-per-request overhead — important when each request holds open an SSE stream for 3–5 seconds.

---

**Q: How would you scale this to 500K+ documents?**

Replace `IndexFlatIP` with `IndexIVFFlat` or HNSW for approximate search. Add a proper vector database (Qdrant, Weaviate) for distributed storage and native metadata filtering. Move BM25 to Elasticsearch for distributed sparse search. The semantic cache and session store already have Redis backends ready. The ingestion pipeline is streaming, so document volume doesn't change its memory profile — just index size and rebuild time.

---

**Q: How does the asymmetric embedding work?**

BGE-base was trained with instruction prefixes on queries but not on passages. At inference, queries are prefixed with `"Represent this sentence for searching relevant passages:"` but document chunks are not. This aligns the query vector to the passage embedding space the way the model was trained — yielding +5–15% recall vs. encoding both symmetrically.

---

**Q: Why keep the system prompt hot-reloadable?**

Prompt tuning is iterative — you discover failure modes during testing and need to add rules quickly. Hot-reloading means changing the prompt file takes effect on the next query without a server restart. In production it also enables incident response: if the model starts misbehaving, a prompt fix is a seconds-level fix, not a deployment.

---

## 10. What Makes This Project Stand Out

| Differentiator | Why it matters |
|----------------|----------------|
| **Domain-specific chunking** | Respects CFR paragraph hierarchy instead of naive sliding windows — a deliberate trade-off justified by the legal structure of regulatory text |
| **No reranker after benchmarking** | Dropped a commonly-used component after empirical testing showed no gain — shows rigour over following tutorials |
| **Three-layer hallucination prevention** | Prompt constraints + heuristic guards + embedding-based grounding verification — each layer catches a different failure mode |
| **Multi-query decomposition** | Addresses a fundamental limitation of single-vector retrieval for multi-concept queries — not a standard RAG feature |
| **Fully local stack** | No API costs, no data egress — critical for pharmaceutical data governance and 21 CFR Part 11 requirements |
| **Dual LLM backend** | Same codebase supports llama.cpp and Ollama — production flexibility without code changes |
| **Semantic cache** | Practical production optimisation turning 60–70% of requests into 5ms responses — demonstrates thinking beyond the demo |
| **Combined scope gate + rewrite** | Two concerns solved in one LLM call — shows cost-aware system design, not just correctness-first thinking |

---

**Stack:** Python 3.11 · FastAPI · FAISS · BAAI/bge-base-en-v1.5 · fastembed BM25 · llama.cpp/Ollama · Pydantic · asyncio · Redis (optional)
