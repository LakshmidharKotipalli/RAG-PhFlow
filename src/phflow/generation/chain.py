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
from phflow.generation.prompts import build_prompt, build_retrieval_query
from phflow.retrieval.filters import build_filter
from phflow.retrieval.hybrid import HybridRetriever


def _parse_used_sources(answer: str, all_sources: list[dict]) -> tuple[str, list[dict]]:
    """
    Extract the SOURCES_USED declaration from the last line of the answer.

    Returns:
        (clean_answer, used_sources)
        - clean_answer: answer with the SOURCES_USED line stripped and any
          residual [SOURCE N] markers removed
        - used_sources: subset of all_sources that the model declared it used;
          falls back to all_sources if the declaration is absent or malformed
    """
    lines = answer.strip().splitlines()

    used_sources = all_sources  # default: show everything
    last_line = lines[-1].strip() if lines else ""

    match = re.match(r"SOURCES_USED:\s*(.+)", last_line, re.IGNORECASE)
    if match:
        declaration = match.group(1).strip()
        lines = lines[:-1]  # remove the declaration line

        if declaration.lower() != "none":
            # Parse "1, 3, 5" → [0, 2, 4] (0-based indices)
            indices = []
            for part in re.split(r"[,\s]+", declaration):
                if part.isdigit():
                    idx = int(part) - 1  # sources are 1-based in the prompt
                    if 0 <= idx < len(all_sources):
                        indices.append(idx)
            if indices:
                seen = set()
                used_sources = [
                    all_sources[i] for i in indices if not (i in seen or seen.add(i))
                ]

    # Reassemble answer and strip any residual [SOURCE N] markers
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

        # Context-aware retrieval query
        retrieval_query = build_retrieval_query(history, query)

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

        # Build prompt: instructions + history + context + question
        context = format_context_block(sections)
        prompt = build_prompt(
            context=context,
            question=query,
            conversation_history=history,
        )

        if stream:
            return self.llm.stream(prompt)

        # Generate
        raw_answer = await self.llm.generate(prompt)
        all_sources = source_summary(sections)

        # Parse declared sources and clean the answer
        answer, used_sources = _parse_used_sources(raw_answer, all_sources)

        return {
            "answer": answer,
            "sources": used_sources,
            "citations_footer": "",
        }
