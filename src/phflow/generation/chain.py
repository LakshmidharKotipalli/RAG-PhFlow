"""
RAG chain: retrieve → rerank → generate.

Wires together the HybridRetriever, prompt builder, and LlamaCppClient
into a single end-to-end call.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from phflow.generation.citations import format_context_block, source_summary
from phflow.generation.llm import LlamaCppClient
from phflow.generation.prompts import build_prompt, build_retrieval_query
from phflow.retrieval.filters import build_filter
from phflow.retrieval.hybrid import HybridRetriever


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
          - sources: list[dict]
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

        # Build prompt: system instructions + history + context + question
        context = format_context_block(sections)
        prompt = build_prompt(
            context=context,
            question=query,
            conversation_history=history,
        )

        if stream:
            return self.llm.stream(prompt)

        # Generate
        answer = await self.llm.generate(prompt)
        sources = source_summary(sections)

        return {
            "answer": answer,
            "sources": sources,
            "citations_footer": "",
        }
