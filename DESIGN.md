# PhFlow — Detailed Design

This document explains every stage of the pipeline from raw XML to final answer, including the rationale behind each design decision.

---

## 1. XML Parsing

**File:** `src/phflow/ingestion/parser.py`

The source data is a single 21.7 MB XML file with 244,264 lines. Loading the full DOM would require ~1 GB of memory, so the parser uses `lxml.iterparse` in streaming mode:

- Events are consumed one `end` event at a time
- `elem.clear()` is called immediately after processing each `DIV8` (section) to release the subtree from memory
- `huge_tree=True` handles deeply nested elements without hitting lxml's default size limits
- `recover=True` tolerates minor XML encoding issues in the source file

**Why streaming?** The eCFR XML is large enough that a full DOM parse would exhaust memory on typical dev machines. Streaming keeps peak memory usage under 200 MB regardless of file size.

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

## 2. Text Cleaning

**File:** `src/phflow/ingestion/cleaner.py`

Each `DIV8` section element is converted to plain text before chunking:

- **Inline tags** (`<I>`, `<E>`, `<SU>`, `<BCAP>`, etc.) are accumulated and joined in-line — they carry emphasis or superscripts but do not break paragraph flow
- **Skip tags** (`<AMDDATE>`, `<EDNOTE>`, `<EFFDNOT>`, etc.) are silently dropped — editorial/amendment metadata not relevant to regulatory content
- **`<TABLE>` elements** are converted to pipe-delimited markdown (`| col | col |`) so their content is searchable and readable by the LLM
- **`<AUTH>` / `<SOURCE>` blocks** are captured separately and tagged `[AUTH]` / `[SOURCE]` so the chunker can route them to authority-type chunks
- Whitespace is normalised: consecutive spaces collapsed, runs of 3+ newlines reduced to 2

---

## 3. Chunking Strategy

**File:** `src/phflow/ingestion/chunker.py`

**Model: Structural parent-child chunking**

CFR sections are hierarchically labelled — `(a)`, `(b)`, `(c)` at the top level, with `(1)`, `(2)`, `(i)`, `(ii)`, `(A)` nested beneath. This structure maps directly to chunking boundaries without needing token-count heuristics.

**Why structural over sliding-window?** Regulatory text is semantically anchored to its paragraph label. A sliding window that cuts mid-paragraph loses the label context (`(b)(2)`) and produces fragments that embed poorly. Structural boundaries preserve the complete regulatory unit.

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

Every chunk is prefixed with a context header:

```
21 CFR § 211.182 — § 211.182 Equipment cleaning and use log.
Part 211: Current Good Manufacturing Practice for Finished Pharmaceuticals | Subpart J: Records and Reports
```

**Why the header?** When the embedding model sees only a fragment of the section, the header ensures it receives the full regulatory address. Without it, short paragraphs like `"(b) Records shall be retained for at least 1 year after the expiration date."` embed with no CFR context, degrading retrieval precision.

**Authority chunks excluded from retrieval**

`chunk_type="authority"` chunks are filtered out at query time. They contain boilerplate like `"Authority: 21 U.S.C. 321, 351…"` which scores high on BM25 for almost any regulatory query and dominates results without contributing substantive content.

**Chunk statistics**

~50,000–55,000 total chunks across ~16,828 sections. Average: 3–4 text/table chunks per section.

---

## 4. Embedding

**File:** `src/phflow/embeddings/encoder.py`

**Model:** `BAAI/bge-base-en-v1.5`
- 768-dimensional dense vectors
- 512-token context window (matches `chunk_max_tokens`)
- Cosine similarity space
- Strong BEIR benchmark performance on factual/regulatory retrieval

**Why BGE-base?** It is the best-performing open model at the 768-dim size point on BEIR regulatory/factual tasks. Larger models (e.g. bge-large) improve recall marginally but double memory and inference time. Smaller models (e.g. MiniLM) show measurable recall degradation on long regulatory text.

**Asymmetric encoding**

BGE-base uses an asymmetric retrieval design: queries and passages are encoded differently to align their vector spaces:

- **Passages (chunks):** encoded with `model.encode(text)` — no prefix
- **Queries:** encoded with `model.encode("Represent this sentence for searching relevant passages: " + query)` — the instruction prefix shifts the query vector toward the passage space

Without this prefix, query vectors and chunk vectors occupy different regions of the 768-dim space, degrading retrieval recall by 5–15% on typical BEIR tasks.

Both query and passage vectors are L2-normalised before storage, so cosine similarity reduces to dot product.

**Query vector cache**

`encode_query()` maintains a module-level LRU cache (256 entries). Repeated or near-identical queries — common during a session — skip model inference entirely. The model is deterministic, so cached results are always correct.

---

## 5. Indexing

**File:** `src/phflow/vectorstore/client.py`, `src/phflow/vectorstore/schema.py`

**Vector index — Dense (FAISS IndexFlatIP)**

```
Distance:       Inner product (equivalent to cosine for L2-normalised vectors)
Dimensions:     768
GPU support:    Auto-detected (faiss-gpu via conda); falls back to CPU
```

**Why FAISS over a vector database?** At ~50K chunks, a dedicated vector DB (Qdrant, Weaviate, Milvus) adds infrastructure complexity with no throughput benefit. FAISS `IndexFlatIP` provides exact nearest-neighbour search with zero approximation error and runs entirely in-process. The index loads in under 2 seconds and the full search + filter cycle takes <100ms on CPU.

**Why `IndexFlatIP` (exact) over `IndexIVFFlat` (approximate)?** At 50K vectors, exact search is fast enough that there is no reason to accept approximation error. ANN indexes only pay off above ~1M vectors.

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
| `section_number` | Keyword | e.g. `"§ 211.182"` |
| `section_node` | Keyword | Unique section address used for parent expansion |
| `section_title` | Keyword | Section heading text |
| `chapter_number` | Keyword | Roman numeral chapter (e.g. `"I"`) |
| `subpart_letter` | Keyword | Subpart letter (e.g. `"B"`) |
| `chunk_type` | Keyword | `"text"`, `"table"`, `"authority"` |
| `latest_amendment_date` | Keyword | ISO date — used for `amended_after` temporal filter |
| `is_removed` | Bool | True if section was repealed — always filtered out |
| `has_table` | Bool | True if section contains a table |
| `substantive_change` | Bool | Marks substantive vs. editorial amendments |

---

## 6. Retrieval

**File:** `src/phflow/retrieval/hybrid.py`

### 6.1 Auto Filter Extraction

**File:** `src/phflow/retrieval/filters.py` → `extract_filters_from_query()`

Before retrieval, the raw query is scanned for explicit filter hints. Any detected values are merged with caller-supplied filters (caller values always take precedence):

| Pattern in query | Auto-set filter | Example |
|---|---|---|
| `"Part 211"` / `"part 211"` | `part_number=211` | "What are Part 211 GMP requirements?" |
| `"§ 211.42"` | `part_number=211` | "What does § 211.42 require?" |
| `"Chapter I"` | `chapter_number="I"` | "Chapter I labeling rules" |
| `"amended after 2022"` | `amended_after="2022-01-01"` | "rules amended after 2022" |

This improves precision for queries that name a specific Part without the user explicitly setting a filter in the API request.

### 6.2 Default Part Scope

When no explicit `part_number` filter is present, the retriever automatically scopes to `SCOPE_PART_NUMBERS` (default: Parts 210 and 211). Generic pharmaceutical GMP queries would otherwise surface results from unrelated parts of Title 21 (food labelling, cosmetics, etc.).

Setting an explicit `part_number` in the request or query text overrides the default scope entirely.

### 6.3 Optional Query Rewrite

**File:** `src/phflow/generation/prompts.py` → `build_retrieval_rewrite_prompt()`

When `ENABLE_QUERY_REWRITE=true` (the default), a short LLM call (`max_tokens=128`) rewrites colloquial workplace language into formal CFR terminology before the retrieval encode step:

- `"fridge temperature for drugs"` → `"storage temperature requirements for pharmaceutical drug products"`
- `"sign off on a digital form"` → `"electronic record signature requirements"`
- `"pallet rejected at dock"` → `"container receipt and acceptance refusal"`

**Why rewrite?** The BGE embedding model was trained on formal text. Operational jargon and CFR terms often occupy different regions of the embedding space. The rewrite bridges this gap without retraining the embedding model.

The rewrite only affects the retrieval query. The original question is retained for prompt assembly so the LLM sees the user's actual words.

### 6.4 Hybrid Search (Dense + BM25 via RRF)

Both retrieval signals are computed and fused client-side using Reciprocal Rank Fusion (RRF):

```
Query
  │
  ├─► encode_query(prefix + query)  →  dense vector  ──► FAISS top-20
  │
  └─► bm25.query_embed(query)       →  sparse vector ──► BM25 top-20
                                                              │
                                              Client-side RRF fusion
                                                    top-10 candidates
```

**Why both dense and BM25?** Dense retrieval excels at semantic similarity but misses exact term matches (e.g. specific CFR section numbers, defined terms). BM25 excels at exact matches but misses paraphrases. Each signal catches what the other misses; fusing them gives consistently higher recall than either alone.

**Why RRF instead of score interpolation?** Dense and BM25 scores exist on incompatible scales. RRF avoids the need to tune a weighting factor `α` by working purely on rank positions:

```
RRF(d) = Σ  1 / (k + rank_i(d))    (k=60)
```

A chunk ranked 1st in both lists scores higher than one ranked 5th in both, regardless of raw score values. This is robust to scale differences and eliminates calibration work.

**Why no reranker?** A cross-encoder reranker was tested and dropped. On regulatory text, RRF fusion of dense + BM25 achieves equivalent P@5 and R@5 to a reranked pipeline, but adds 300–800ms latency per query. The gain did not justify the cost.

### 6.5 Metadata Filtering

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

The `is_removed=False` filter is always active. Repealed sections must never surface as valid regulatory guidance.

---

## 7. Parent Expansion

**File:** `src/phflow/retrieval/hybrid.py` → `HybridRetriever.search()`

After RRF fusion, the top-N results are individual *chunks* (one paragraph group each). The LLM needs the *full section* for context — a single `(b)` sub-paragraph without its surrounding `(a)` and `(c)` is often ambiguous or incomplete.

For each unique `section_node` in the top-N results:

1. Scroll all chunks sharing that `section_node` from FAISS metadata
2. Concatenate their `chunk_text` in `chunk_index` order
3. Strip the repeated section header from subsequent chunks (only the first is kept)
4. Replace the single matched chunk with the full assembled section text

If two top-N chunks share the same section, they are deduplicated — the section is fetched once.

**Why expand?** Embedding a full section as a single chunk would exceed the 512-token embedding window, losing the tail paragraphs. Parent expansion solves the retrieval/context-length tension: embed small, retrieve full.

---

## 8. Query Optimisation

**File:** `src/phflow/generation/prompts.py`, `src/phflow/generation/chain.py`

**Adaptive history window**

The prompt includes conversation history to resolve follow-up references. The window adapts to the query:

- **Default (`HISTORY_WINDOW = 0`):** no prior turns included — eliminates hallucination risk on fresh queries
- **Expanded (2 turns / 4 messages):** automatically activated when the query contains a back-reference pronoun or phrase (`"that"`, `"it"`, `"its"`, `"same"`, `"the section"`, `"the requirement"`, `"previous"`, etc.)

**Why default to 0?** Early testing showed that including prior conversation turns on fresh queries — even unrelated ones — caused the LLM to blend the current answer with prior topics. The adaptive trigger activates history only when the query syntactically requires it.

**Retrieval query isolation**

Prior conversation turns are never concatenated into the retrieval query. Mixing prior turns into the retrieval query pulls the query vector toward the previous topic, degrading precision for the current question.

---

## 9. Prompt Assembly

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

**Why `[SOURCE N]` labels?** They allow the model to declare which sources it actually used, rather than returning all retrieved sections as citations. The `SOURCES_USED:` declaration maps back to the actual sections so the API can return only the genuinely used sources, reducing noise in the client response.

---

## 10. Generation and Output Cleaning

**File:** `src/phflow/generation/llm.py`, `src/phflow/generation/chain.py`

### 10.1 LLM Client

The `LlamaCppClient` posts to the llama.cpp `/completion` endpoint with these key parameters:

```json
{
  "temperature":       0.1,
  "repeat_penalty":    1.1,
  "repeat_last_n":     256,
  "frequency_penalty": 0.1,
  "stop":              ["<|im_end|>", "</s>", "<|eot_id|>", "INSTRUCTIONS:", "CONTEXT:", "QUESTION:"]
}
```

**Temperature `0.1`** — near-deterministic. Regulatory answers must be factual, not creative.

**Repeat penalty** — `repeat_penalty=1.1` combined with `frequency_penalty=0.1` prevents two observed failure modes:
1. The `SOURCES_USED:` line looping indefinitely
2. The model echoing the question + context back as its answer body

**Stop sequences** — EOS and role tokens across major model families plus prompt section labels (`INSTRUCTIONS:`, `CONTEXT:`, `QUESTION:`) stop generation cleanly and prevent the model from generating a second prompt-answer pair.

### 10.2 Output Cleaning

After generation, `_clean()` applies multiple passes:

1. **`<think>` block removal** — strips reasoning blocks emitted by chain-of-thought models (Qwen3, DeepSeek-R1)
2. **EOS token stripping** — leaked `<|im_end|>`, `</s>`, etc. are stripped
3. **Section-label stripping** — prompt section headers echoed by the model are stripped

### 10.3 SOURCES_USED Parsing

`_parse_used_sources()` in `chain.py`:

- Scans **all lines** for `SOURCES_USED:` and takes the **last** occurrence. The last is preferred because without repeat_penalty some models generate two complete answers; the second is typically better.
- Accepts both `SOURCES_USED:` and `SOURCES USED:` (space variant emitted by some models)
- Parses 1-based source indices and maps them to the actual `source_summary` entries
- Strips the `SOURCES_USED:` line and everything after it from the answer body

---

## 11. Answer Integrity Guards

**File:** `src/phflow/generation/chain.py`

After parsing, the answer passes through several guards. Any failure replaces the answer with the standard insufficient-context message and clears sources:

| Guard | Condition | Failure mode caught |
|---|---|---|
| **No sections** | `not sections` after retrieval | Retrieval failure — LLM never called |
| **Empty answer** | `not answer` | Model returned nothing after cleaning |
| **Bare CFR header** | Answer matches a section title/number pattern | Model echoed a context block label as its entire answer |
| **Too short** | `len(answer) < 60` | Parametric one-liner from training memory |
| **Prompt echo** | `"CONTEXT:"` / `"QUESTION:"` / `"INSTRUCTIONS:"` in answer | Model reproduced prompt structure in its answer body |
| **Template placeholder** | `"[fill in"` / `"[insert"` / `"[answer here"` in answer | Model produced a fill-in-the-blank stub |
| **Reasoning leak** | `"<think>"` in answer | Chain-of-thought reasoning leaked into output |

**The no-sections short-circuit is critical.** When all retrieved chunks are filtered out, the LLM must not be called. Without this guard, the LLM receives `"No relevant regulatory sections found."` as its context and hallucinates a confident answer from training memory.

---

## 12. Session Management

**File:** `src/phflow/api/session_store.py`

Sessions persist conversation history (`user` + `assistant` message pairs) across turns. Two backends are available:

| Backend | Activation | Behaviour |
|---|---|---|
| `InMemorySessionStore` | Default (no config needed) | Module-level dict; sessions lost on restart |
| `RedisSessionStore` | Set `REDIS_URL=redis://localhost:6379` | JSON-serialized in Redis with 24-hour TTL; sessions survive restarts |

The API selects the backend automatically at startup. If `REDIS_URL` is set but Redis is unreachable, it falls back to in-memory silently. Both backends trim history to `SESSION_MAX_MESSAGES` (default: 100) to prevent unbounded memory growth.

---

## 13. Per-Query Debug Logging

**File:** `src/phflow/utils/query_logger.py`

Every query writes one timestamped JSON file to `query_logs/`:

```
query_logs/
└── 20260309_141457_691117__What_are_the_GMP_requirements_for.json
```

Each file captures (in order):
1. Original query
2. Retrieval query (after rewrite, if enabled)
3. Auto-extracted filters
4. Applied scope (if default scope was used)
5. Retrieval candidates: section number, title, RRF score (top-N)
6. Sections passed to LLM: section number, title, RRF score, full_text char count
7. Assembled prompt — stored as an array of lines for readability
8. Raw LLM output — stored as an array of lines
9. Final answer — stored as an array of lines
10. Sources declared by the LLM

**Why JSON?** Structured JSON enables aggregate analysis (e.g. "what % of queries had 0 sections after retrieval?") and is readable in any JSON viewer. Storing prompt and output as arrays avoids escaping issues with multi-line strings.
