"""
RAG chain: retrieve → rerank → generate.

Wires together the HybridRetriever, prompt builder, and LlamaCppClient
into a single end-to-end call.
"""

from __future__ import annotations

import re
from collections.abc import AsyncIterator

from phflow.settings import settings
from phflow.generation.citations import format_context_block, source_summary
from phflow.generation.llm import LlamaCppClient
from phflow.generation.prompts import (
    INSUFFICIENT_MESSAGE,
    build_prompt,
    build_retrieval_query,
    build_retrieval_rewrite_prompt,
)
from phflow.retrieval.filters import build_filter, extract_filters_from_query
from phflow.retrieval.hybrid import HybridRetriever
from phflow.utils.query_logger import QueryLogger



# Answers shorter than this are almost certainly hallucinated one-liners
# (e.g. "2–8°C", "✘ No, you cannot.") that bypassed the context-only rule.
# The _INSUFFICIENT message itself is 78 chars, so 60 is a safe lower bound.
_MIN_ANSWER_LENGTH = 60

# Prompt-section labels that must never appear inside a legitimate answer.
# Their presence indicates the model echoed the prompt back, or is deliberating
# inline ("Wait, I need to check…" style reasoning leaking into the output).
_PROMPT_ECHO_SIGNALS = (
    "CONTEXT:", "QUESTION:", "INSTRUCTIONS:", "CONVERSATION HISTORY:",
    "Let me re-read", "Wait, I need", "Actually, wait",
    # Phrases unique to our system prompt rule 6 — if any of these appear in
    # the answer the model echoed the instructions back instead of answering.
    "formal prose paragraph",
    "Do not use bullet points",
    "Do not restate the question",
    "CONTEXT block",
    "source labels (e.g.",
)

# Substrings that indicate the model produced a fill-in-the-blank placeholder
# instead of a real answer (e.g. "by [fill in the blank]").
_TEMPLATE_SIGNALS = ("[fill in", "[insert", "[answer here", "[provide")


def _parse_used_sources(answer: str, all_sources: list[dict]) -> tuple[str, list[dict]]:
    """
    Find the SOURCES_USED declaration in the answer.

    Scans all lines and takes the LAST occurrence. This is better than taking
    the first because:
    - Without repeat_penalty the model sometimes generates two complete answers;
      the second (last) answer is typically more detailed and correct.
    - Repetition loops all declare the same sources, so last == first anyway.

    Returns:
        (clean_answer, used_sources)
        - clean_answer: answer with the last declaration line and everything
          after it stripped, and residual [SOURCE N] markers removed.
        - used_sources: subset of all_sources the model declared it used.
          Defaults to [] (empty) if the declaration is absent.
    """
    lines = answer.strip().splitlines()
    used_sources: list[dict] = []
    declaration_line_idx: int | None = None

    # Scan ALL lines, keeping the LAST matching SOURCES_USED line.
    for i, line in enumerate(lines):
        m = re.match(r"SOURCES[_ ]USED\s*:\s*(.+)", line.strip(), re.IGNORECASE)
        if m:
            declaration_line_idx = i          # update on every match → ends at last
            declaration = m.group(1).strip()
            used_sources = []                 # reset for each new declaration
            if declaration.lower() != "none":
                seen: set[int] = set()
                for part in re.split(r"[,\s]+", declaration):
                    if part.isdigit():
                        idx = int(part) - 1   # 1-based in prompt → 0-based
                        if 0 <= idx < len(all_sources) and idx not in seen:
                            seen.add(idx)
                            used_sources.append(all_sources[idx])
            # no break — continue to find the last occurrence

    # Keep only the lines before the declaration
    if declaration_line_idx is not None:
        lines = lines[:declaration_line_idx]

    # Reassemble and remove any residual source-label markers.
    # Catches both bracketed form "[SOURCE 1]" and plain "Source 1" / "source 1"
    # that some models emit despite the system prompt prohibiting them.
    clean = "\n".join(lines).strip()
    clean = re.sub(r"\[SOURCE\s*\d+\]", "", clean)
    clean = re.sub(r"\bSource\s+\d+\b", "", clean, flags=re.IGNORECASE)
    clean = re.sub(r"\(\s*\)", "", clean)   # remove empty parens left by stripping
    clean = clean.strip()

    return clean, used_sources


class RAGChain:
    def __init__(self, retriever: HybridRetriever, llm: LlamaCppClient) -> None:
        self.retriever = retriever
        self.llm = llm

    async def run(
        self,
        query: str,
        mode: str = "qa",
        conversation_history: list[dict] | None = None,
        part_number: int | None = None,
        chapter_number: str | None = None,
        amended_after: str | None = None,
        expand_to_parent: bool = True,
        stream: bool = False,
    ) -> dict | AsyncIterator[str]:
        """
        Run the full RAG pipeline for a query.

        Returns a dict with keys:
          - answer: str
          - sources: list[dict]  (only sources the model declared it used)
          - citations_footer: str
        Or an async iterator of token strings if stream=True.
        """
        history = conversation_history or []
        qlog = QueryLogger(query)
        qlog.log("ORIGINAL QUERY", query)
        pipeline_steps: list[str] = []

        clean_query = query

        # Stage 1: Query rewrite
        if settings.enable_query_rewrite:
            pipeline_steps.append("Rewriting query to CFR terminology")
            rewrite_prompt = build_retrieval_rewrite_prompt(clean_query, conversation_history=None)
            rewritten = await self.llm.correct_query(rewrite_prompt)
            retrieval_query = rewritten if rewritten else build_retrieval_query(history, clean_query)
        else:
            retrieval_query = build_retrieval_query(history, clean_query)
        qlog.log("RETRIEVAL QUERY", retrieval_query)

        # Stage 2: Filter extraction
        filter_kwargs = extract_filters_from_query(
            clean_query,
            {
                "part_number": part_number,
                "section_number": None,
                "chapter_number": chapter_number,
                "amended_after": amended_after,
            },
        )
        if filter_kwargs != {"part_number": part_number, "section_number": None, "chapter_number": chapter_number, "amended_after": amended_after}:
            qlog.log("AUTO-EXTRACTED FILTERS", filter_kwargs)

        explicit_part = filter_kwargs["part_number"]
        scope = settings.scope_part_numbers if not explicit_part else None
        if scope:
            qlog.log("APPLIED SCOPE", {"scope_part_numbers": scope})

        search_filter = build_filter(
            part_number=explicit_part,
            part_numbers=scope or None,
            section_number=filter_kwargs.get("section_number"),
            chapter_number=filter_kwargs["chapter_number"],
            amended_after=filter_kwargs["amended_after"],
            exclude_removed=True,
        )

        # Stage 3: Retrieval
        pipeline_steps.append("Retrieving relevant regulatory sections")
        sections = self.retriever.search(
            query=retrieval_query,
            search_filter=search_filter,
            expand_to_parent=expand_to_parent,
            logger=qlog,
        )

        qlog.log("SECTIONS PASSED TO LLM", [
            {
                "section_number": s.get("section_number", ""),
                "section_title": s.get("section_title", ""),
                "rrf_score": s.get("score", 0.0),
                "full_text_chars": len(s.get("full_text", "")),
            }
            for s in sections
        ])

        # Stage 4: Reranking result note
        pipeline_steps.append(
            f"Reranked to {len(sections)} section{'s' if len(sections) != 1 else ''}"
            if sections else "No relevant sections found"
        )

        if not sections:
            qlog.log("FINAL ANSWER", INSUFFICIENT_MESSAGE)
            qlog.log("SOURCES DECLARED BY LLM", [])
            qlog.save_json()
            return {"answer": INSUFFICIENT_MESSAGE, "sources": [], "pipeline_steps": pipeline_steps}

        # Stage 5: Prompt assembly
        pipeline_steps.append("Assembling prompt")
        context = format_context_block(sections)
        prompt = build_prompt(
            context=context,
            question=clean_query,
            conversation_history=history,
        )
        qlog.log("ASSEMBLED PROMPT", prompt)

        # Stage 6: LLM generation (streaming returns the generator; steps sent separately)
        if stream:
            pipeline_steps.append("Generating answer")
            qlog.save_json()

            async def _stream_with_status():
                for step in pipeline_steps:
                    yield f"[PIPELINE] {step}\n"
                async for token in self.llm.stream(prompt):
                    yield token

            return _stream_with_status()

        pipeline_steps.append("Generating answer with LLM")
        raw_answer = await self.llm.generate(prompt)
        qlog.log("RAW LLM ANSWER", raw_answer)

        all_sources = source_summary(sections)
        answer, used_sources = _parse_used_sources(raw_answer, all_sources)

        # Strip question echo
        q_stripped = clean_query.strip()
        if answer.startswith(q_stripped):
            answer = answer[len(q_stripped):].strip()

        # Check if the answer is just an echoed section header
        is_bare_header = False
        answer_stripped = answer.strip()
        answer_lower = answer_stripped.lower()
        
        for s in sections:
            title = s.get("section_title", "").strip().lower()
            num = s.get("section_number", "").strip().lower()
            clean_title = title.replace(num, "").strip() if num else title
            
            # Reconstruct the headers the LLM saw in the context block
            header_candidates = [
                title,
                f"21 cfr {num} \u2014 {title}",
                f"21 cfr {num} - {title}",
                f"21 cfr {num} \u2014 {clean_title}",
                f"21 cfr {num} - {clean_title}",
                f"21 cfr {title}"
            ]
            
            for candidate in header_candidates:
                # If the answer perfectly mimics the header or is a substring of it
                if answer_lower == candidate or (answer_lower in candidate and len(answer_lower) > 20):
                    is_bare_header = True
                    break
            if is_bare_header:
                break

        # Degenerate answer guards
        if (
            not answer
            or is_bare_header
            or len(answer_stripped) < _MIN_ANSWER_LENGTH
            or any(sig in answer for sig in _PROMPT_ECHO_SIGNALS)
            or any(sig in answer_lower for sig in _TEMPLATE_SIGNALS)
            or "<think>" in answer_lower
        ):
            answer = INSUFFICIENT_MESSAGE
            used_sources = []

        qlog.log("FINAL ANSWER", answer)
        qlog.log("SOURCES DECLARED BY LLM", [
            s.get("section_number", "") for s in used_sources
        ])
        qlog.save_json()

        return {
            "answer": answer,
            "sources": used_sources,
            "pipeline_steps": pipeline_steps,
        }
