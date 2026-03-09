from __future__ import annotations

from fastapi import APIRouter, Depends

from phflow.api.dependencies import get_chain
from phflow.api.schemas import ComplianceRequest, ComplianceResponse, SourceResult
from phflow.generation.chain import RAGChain

router = APIRouter(prefix="/api/v1", tags=["compliance"])


@router.post("/compliance/check", response_model=ComplianceResponse)
async def compliance_check(
    req: ComplianceRequest,
    chain: RAGChain = Depends(get_chain),
) -> ComplianceResponse:
    result = await chain.run(
        query=req.scenario,
        mode="compliance",
        part_number=req.filters.part_number,
        chapter_number=req.filters.chapter_number,
        amended_after=req.filters.amended_after,
        expand_to_parent=True,
        stream=False,
    )

    sources = [
        SourceResult(
            section_number=s.get("section_number", ""),
            section_title=s.get("section_title", ""),
            section_node=s.get("section_node", ""),
            part_number=s.get("part_number"),
            part_name=s.get("part_name", ""),
            latest_amendment_date=s.get("latest_amendment_date", ""),
            score=s.get("score", 0.0),
            full_text=s.get("full_text", ""),
        )
        for s in result.get("sources", [])
    ]

    return ComplianceResponse(
        answer=result["answer"],
        sources=sources,
        citations_footer=result.get("citations_footer", ""),
    )
