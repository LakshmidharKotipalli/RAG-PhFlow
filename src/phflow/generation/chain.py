"""
RAG chain: retrieve → rerank → generate.

Wires together the HybridRetriever, prompt builder, and LlamaCppClient
into a single end-to-end call.
"""

from __future__ import annotations

import re
from collections.abc import AsyncIterator

from phflow.generation.citations import format_context_block, source_summary
from phflow.generation.llm import LlamaCppClient
from phflow.generation.prompts import build_correction_prompt, build_prompt, build_retrieval_query
from phflow.retrieval.filters import build_filter
from phflow.retrieval.hybrid import HybridRetriever


# Matches a bare CFR section header echo, e.g. "21 CFR § 117.3 — § 117.3 Definitions."
# These are degenerate answers where the LLM echoed a context block header.
_BARE_HEADER_RE = re.compile(
    r"^21\s+CFR\s+§?\s*[\d.]+(\s*[\u2014\-]\s*.+)?\.?$",
    re.IGNORECASE,
)

_INSUFFICIENT = (
    "The provided context does not contain sufficient information "
    "to answer this question."
)


def _parse_used_sources(answer: str, all_sources: list[dict]) -> tuple[str, list[dict]]:
    """
    Find the SOURCES_USED declaration in the answer (scans bottom-up so it is
    found even when the LLM adds trailing blank lines or puts it one line above
    the last line).

    Returns:
        (clean_answer, used_sources)
        - clean_answer: answer with the declaration line stripped and any
          residual [SOURCE N] markers removed
        - used_sources: subset of all_sources the model declared it used.
          Defaults to [] (empty) if the declaration is absent — callers should
          not fall back to showing everything when the LLM skips it.
    """
    lines = answer.strip().splitlines()
    used_sources: list[dict] = []   # default: show nothing
    declaration_line_idx: int | None = None

    # Scan from the bottom up to find the declaration
    for i in range(len(lines) - 1, -1, -1):
        m = re.match(r"SOURCES_USED:\s*(.+)", lines[i].strip(), re.IGNORECASE)
        if m:
            declaration_line_idx = i
            declaration = m.group(1).strip()
            if declaration.lower() != "none":
                seen: set[int] = set()
                for part in re.split(r"[,\s]+", declaration):
                    if part.isdigit():
                        idx = int(part) - 1   # 1-based in prompt → 0-based
                        if 0 <= idx < len(all_sources) and idx not in seen:
                            seen.add(idx)
                            used_sources.append(all_sources[idx])
            break

    # Strip everything from the declaration line onwards
    if declaration_line_idx is not None:
        lines = lines[:declaration_line_idx]

    # Reassemble and remove any residual [SOURCE N] markers
    clean = "\n".join(lines).strip()
    clean = re.sub(r"\[SOURCE\s*\d+\]", "", clean).strip()
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

        # Correct spelling and rephrase colloquial input before retrieval
        correction_prompt = build_correction_prompt(query)
        corrected = await self.llm.correct_query(correction_prompt)
        clean_query = corrected if corrected else query  # fallback to original

        # Context-aware retrieval query (uses corrected query)
        retrieval_query = build_retrieval_query(history, clean_query)

        # Build metadata filter
        qdrant_filter = build_filter(
            part_number=part_number,
            chapter_number=chapter_number,
            amended_after=amended_after,
            exclude_removed=True,
        )

        # Retrieve + rerank
        sections = self.retriever.search(
            query=retrieval_query,
            qdrant_filter=qdrant_filter,
            expand_to_parent=expand_to_parent,
        )

        # Build prompt: instructions + history + context + corrected question
        context = format_context_block(sections)
        prompt = build_prompt(
            context=context,
            question=clean_query,
            conversation_history=history,
        )

        if stream:
            return self.llm.stream(prompt)

        # Generate
        raw_answer = await self.llm.generate(prompt)
        all_sources = source_summary(sections)

        # Parse declared sources and clean the answer
        answer, used_sources = _parse_used_sources(raw_answer, all_sources)

        # Catch degenerate answers: empty or bare CFR section header echoes
        if not answer or _BARE_HEADER_RE.match(answer.strip()):
            answer = _INSUFFICIENT
            used_sources = []

        return {
            "answer": answer,
            "sources": used_sources,
        }
