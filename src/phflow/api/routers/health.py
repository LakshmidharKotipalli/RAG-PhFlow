from __future__ import annotations

from fastapi import APIRouter, Request

from phflow.api.schemas import ComponentHealth, HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health_check(request: Request) -> HealthResponse:
    components: dict[str, ComponentHealth] = {}

    # Check Qdrant
    try:
        qdrant = request.app.state.qdrant
        info = qdrant.get_collections()
        components["qdrant"] = ComponentHealth(status="ok")
    except Exception as exc:
        components["qdrant"] = ComponentHealth(status="error", detail=str(exc))

    # Check llama.cpp server
    try:
        llm = request.app.state.llm
        ok = await llm.health()
        components["llama_cpp"] = ComponentHealth(
            status="ok" if ok else "error",
            detail="" if ok else "llama.cpp server unreachable or model not loaded",
        )
    except Exception as exc:
        components["llama_cpp"] = ComponentHealth(status="error", detail=str(exc))

    # Encoder is always loaded synchronously at startup — mark ok if we got here
    components["encoder"] = ComponentHealth(status="ok")

    overall = (
        "ok"
        if all(c.status == "ok" for c in components.values())
        else "degraded"
    )
    return HealthResponse(status=overall, components=components)
