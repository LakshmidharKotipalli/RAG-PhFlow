from __future__ import annotations

import uuid
from datetime import datetime, timezone

import httpx
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from qdrant_client.http.exceptions import ResponseHandlingException

from phflow.api.dependencies import get_chain
from phflow.api.schemas import ChatRequest, ChatResponse, SourceResult
from phflow.generation.chain import RAGChain
from phflow.generation.citations import format_citations_footer, source_summary

router = APIRouter(prefix="/api/v1", tags=["chat"])

# In-memory session store (dev mode)
# For production: replace with Redis
_sessions: dict[str, dict] = {}


def _get_or_create_session(session_id: str | None) -> tuple[str, dict]:
    if session_id and session_id in _sessions:
        return session_id, _sessions[session_id]
    sid = str(uuid.uuid4())
    _sessions[sid] = {
        "session_id": sid,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "messages": [],
    }
    return sid, _sessions[sid]


async def _handle_chat(
    req: ChatRequest,
    session_id: str | None,
    chain: RAGChain,
) -> ChatResponse:
    sid, session = _get_or_create_session(session_id)
    history = session["messages"]

    try:
        result = await chain.run(
            query=req.message,
            mode=req.mode,
            conversation_history=history,
            part_number=req.filters.part_number,
            chapter_number=req.filters.chapter_number,
            amended_after=req.filters.amended_after,
            expand_to_parent=True,
            stream=False,
        )
    except (ResponseHandlingException, httpx.ConnectError):
        raise HTTPException(
            status_code=503,
            detail="A backend service is unavailable. Ensure Qdrant (docker compose up -d) and llama-server are running.",
        )

    # Update session history
    session["messages"].append({"role": "user", "content": req.message})
    session["messages"].append({"role": "assistant", "content": result["answer"]})

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

    return ChatResponse(
        session_id=sid,
        answer=result["answer"],
        sources=sources,
        citations_footer=result.get("citations_footer", ""),
        mode=req.mode,
    )


@router.post("/chat", response_model=ChatResponse)
async def new_chat(
    req: ChatRequest,
    chain: RAGChain = Depends(get_chain),
):
    if req.stream:
        # Streaming: return SSE-style response
        result = await chain.run(
            query=req.message,
            mode=req.mode,
            stream=True,
            part_number=req.filters.part_number,
            chapter_number=req.filters.chapter_number,
            amended_after=req.filters.amended_after,
        )

        async def token_generator():
            async for token in result:
                yield token

        return StreamingResponse(token_generator(), media_type="text/plain")

    return await _handle_chat(req, session_id=None, chain=chain)


@router.post("/chat/{session_id}/message", response_model=ChatResponse)
async def continue_chat(
    session_id: str,
    req: ChatRequest,
    chain: RAGChain = Depends(get_chain),
):
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")

    if req.stream:
        result = await chain.run(
            query=req.message,
            mode=req.mode,
            conversation_history=_sessions[session_id]["messages"],
            stream=True,
            part_number=req.filters.part_number,
            chapter_number=req.filters.chapter_number,
        )

        async def token_generator():
            async for token in result:
                yield token

        return StreamingResponse(token_generator(), media_type="text/plain")

    return await _handle_chat(req, session_id=session_id, chain=chain)
