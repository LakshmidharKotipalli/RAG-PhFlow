from __future__ import annotations

import logging

from fastapi import APIRouter, Request

from phflow.api.schemas import ComponentHealth, HealthResponse

logger = logging.getLogger(__name__)

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health_check(request: Request) -> HealthResponse:
    components: dict[str, ComponentHealth] = {}

    # Check FAISS store
    try:
        store = request.app.state.store
        n = store.total_points
        components["faiss"] = ComponentHealth(
            status="ok" if n > 0 else "error",
            detail=f"{n} vectors loaded" if n > 0 else "FAISS index is empty",
        )
    except Exception as exc:
        logger.warning("FAISS health check failed: %s", exc)
        components["faiss"] = ComponentHealth(status="error", detail="FAISS store unavailable")

    # Check llama.cpp server
    try:
        llm = request.app.state.llm
        ok = await llm.health()
        components["llama_cpp"] = ComponentHealth(
            status="ok" if ok else "error",
            detail="" if ok else "llama.cpp server unreachable or model not loaded",
        )
    except Exception as exc:
        logger.warning("llama.cpp health check failed: %s", exc)
        components["llama_cpp"] = ComponentHealth(status="error", detail="llama.cpp server unreachable")

    # Encoder is always loaded synchronously at startup — mark ok if we got here
    components["encoder"] = ComponentHealth(status="ok")

    overall = (
        "ok"
        if all(c.status == "ok" for c in components.values())
        else "degraded"
    )
    return HealthResponse(status=overall, components=components)
