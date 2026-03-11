from __future__ import annotations

import re

import httpx
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from phflow.api.dependencies import get_chain, get_session_store
from phflow.api.routers.search import _confidence_label
from phflow.api.schemas import ChatRequest, ChatResponse, SourceResult
from phflow.api.session_store import SessionStore
from phflow.generation.chain import RAGChain

_UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.I)

router = APIRouter(prefix="/api/v1", tags=["chat"])


async def _handle_chat(
    req: ChatRequest,
    session_id: str | None,
    chain: RAGChain,
    store: SessionStore,
) -> ChatResponse:
    if session_id and store.exists(session_id):
        sid, session = session_id, store.get(session_id)
    else:
        sid, session = store.create()

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
    except httpx.ReadTimeout:
        raise HTTPException(
            status_code=504,
            detail=(
                "The LLM did not respond in time. The model may be overloaded or "
                "the prompt is too large. Try a shorter question, or increase "
                "LLAMA_CPP_NUM_CTX / raise the server timeout."
            ),
        )
    except (httpx.ConnectError, ConnectionError, OSError):
        raise HTTPException(
            status_code=503,
            detail="A backend service is unavailable. Ensure llama-server is running.",
        )

    # Update session history and persist
    session["messages"].append({"role": "user", "content": req.message})
    session["messages"].append({"role": "assistant", "content": result["answer"]})
    store.save(sid, session)

    sources = [
        SourceResult(
            section_number=s.get("section_number", ""),
            section_title=s.get("section_title", ""),
            section_node=s.get("section_node", ""),
            part_number=s.get("part_number"),
            part_name=s.get("part_name", ""),
            latest_amendment_date=s.get("latest_amendment_date", ""),
            confidence=s.get("score", 0.0),
            confidence_label=_confidence_label(s.get("score", 0.0)),
            full_text=s.get("full_text", ""),
        )
        for s in result.get("sources", [])
    ]

    return ChatResponse(
        session_id=sid,
        answer=result["answer"],
        sources=sources,
        mode=req.mode,
        pipeline_steps=result.get("pipeline_steps", []),
    )


@router.post("/chat", response_model=ChatResponse)
async def new_chat(
    req: ChatRequest,
    chain: RAGChain = Depends(get_chain),
    store: SessionStore = Depends(get_session_store),
):
    if req.stream:
        sid, session = store.create()
        result = await chain.run(
            query=req.message,
            mode=req.mode,
            conversation_history=session["messages"],
            stream=True,
            part_number=req.filters.part_number,
            chapter_number=req.filters.chapter_number,
            amended_after=req.filters.amended_after,
        )
        session["messages"].append({"role": "user", "content": req.message})

        async def token_generator():
            accumulated: list[str] = []
            async for token in result:
                accumulated.append(token)
                yield token
            # Persist assistant response after stream completes
            session["messages"].append({"role": "assistant", "content": "".join(accumulated)})
            store.save(sid, session)

        return StreamingResponse(
            token_generator(),
            media_type="text/plain",
            headers={"X-Session-Id": sid},
        )

    return await _handle_chat(req, session_id=None, chain=chain, store=store)


@router.post("/chat/{session_id}/message", response_model=ChatResponse)
async def continue_chat(
    session_id: str,
    req: ChatRequest,
    chain: RAGChain = Depends(get_chain),
    store: SessionStore = Depends(get_session_store),
):
    if not _UUID_RE.match(session_id):
        raise HTTPException(status_code=400, detail="Invalid session ID format")
    if not store.exists(session_id):
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")

    if req.stream:
        session = store.get(session_id)
        result = await chain.run(
            query=req.message,
            mode=req.mode,
            conversation_history=session["messages"],
            stream=True,
            part_number=req.filters.part_number,
            chapter_number=req.filters.chapter_number,
            amended_after=req.filters.amended_after,
        )
        session["messages"].append({"role": "user", "content": req.message})

        async def token_generator():
            accumulated: list[str] = []
            async for token in result:
                accumulated.append(token)
                yield token
            session["messages"].append({"role": "assistant", "content": "".join(accumulated)})
            store.save(session_id, session)

        return StreamingResponse(
            token_generator(),
            media_type="text/plain",
            headers={"X-Session-Id": session_id},
        )

    return await _handle_chat(req, session_id=session_id, chain=chain, store=store)
