from __future__ import annotations

import time

from fastapi import APIRouter, Depends, HTTPException

from phflow.api.dependencies import get_retriever
from phflow.api.schemas import PartInfo, SearchRequest, SearchResponse, SourceResult
from phflow.settings import settings
from phflow.retrieval.filters import build_filter
from phflow.retrieval.hybrid import HybridRetriever

router = APIRouter(prefix="/api/v1", tags=["search"])


def _confidence_label(score: float) -> str:
    if score >= 0.70:
        return "high"
    if score >= 0.40:
        return "medium"
    return "low"


@router.post("/search", response_model=SearchResponse)
async def search(
    req: SearchRequest,
    retriever: HybridRetriever = Depends(get_retriever),
) -> SearchResponse:
    start = time.perf_counter()

    # Apply the same scope logic as the chat endpoint:
    # explicit part_number from request wins; otherwise fall back to configured scope.
    explicit_part = req.filters.part_number
    scope = settings.scope_part_numbers if not explicit_part else None

    search_filter = build_filter(
        part_number=explicit_part,
        part_numbers=scope or None,
        chapter_number=req.filters.chapter_number,
        subpart_letter=req.filters.subpart_letter,
        chunk_types=req.filters.chunk_types,
        amended_after=req.filters.amended_after,
        exclude_removed=True,
    )

    sections = retriever.search(
        query=req.query,
        search_filter=search_filter,
        expand_to_parent=req.expand_to_parent,
        top_n=req.top_k,
    )

    elapsed_ms = (time.perf_counter() - start) * 1000

    results = [
        SourceResult(
            section_number=s.get("section_number", ""),
            section_title=s.get("section_title", ""),
            section_node=s.get("section_node", ""),
            part_number=s.get("metadata", {}).get("part_number"),
            part_name=s.get("metadata", {}).get("part_name", ""),
            latest_amendment_date=s.get("metadata", {}).get("latest_amendment_date", ""),
            confidence=s.get("score", 0.0),
            confidence_label=_confidence_label(s.get("score", 0.0)),
            full_text=s.get("full_text", ""),
        )
        for s in sections
    ]

    return SearchResponse(results=results, total=len(results), query_time_ms=round(elapsed_ms, 1))


@router.get("/sections/{section_node:path}", response_model=dict)
async def get_section(
    section_node: str,
    retriever: HybridRetriever = Depends(get_retriever),
) -> dict:
    chunks = retriever.store.scroll_by_node(section_node)
    if not chunks:
        raise HTTPException(status_code=404, detail=f"Section '{section_node}' not found")

    first = chunks[0]["payload"]
    full_text = "\n\n".join(c["payload"].get("chunk_text", "") for c in chunks)

    return {
        "section_node": section_node,
        "section_number": first.get("section_number", ""),
        "section_title": first.get("section_title", ""),
        "full_text": full_text,
        "authority_text": first.get("authority_text", ""),
        "source_text": first.get("source_text", ""),
        "cita_text": first.get("cita_text", ""),
        "part_number": first.get("part_number"),
        "part_name": first.get("part_name", ""),
        "subpart_letter": first.get("subpart_letter", ""),
        "subpart_name": first.get("subpart_name", ""),
        "latest_amendment_date": first.get("latest_amendment_date", ""),
        "is_removed": first.get("is_removed", False),
        "has_table": first.get("has_table", False),
        "total_chunks": first.get("total_chunks", len(chunks)),
    }


@router.get("/parts", response_model=list[PartInfo])
async def list_parts(retriever: HybridRetriever = Depends(get_retriever)) -> list[PartInfo]:
    """Return distinct parts for use in filter UI dropdowns."""
    seen_parts: dict[int, PartInfo] = {}

    all_results = retriever.store.scroll(
        filter_fn=lambda p: p.get("chunk_index") == 0 and p.get("chunk_type") == "text",
        limit=100_000,
    )

    for item in all_results:
        p = item["payload"]
        pnum = p.get("part_number")
        if pnum and pnum not in seen_parts:
            seen_parts[pnum] = PartInfo(
                part_number=pnum,
                part_name=p.get("part_name", ""),
                chapter_number=p.get("chapter_number", ""),
                chapter_name=p.get("chapter_name", ""),
            )

    return sorted(seen_parts.values(), key=lambda x: x.part_number)
