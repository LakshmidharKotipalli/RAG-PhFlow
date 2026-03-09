"""
Chainlit UI for PhFlow — FDA Regulatory Intelligence.

Run with: chainlit run ui/app.py

Connects to the FastAPI backend at http://localhost:8000.
"""

from __future__ import annotations

import httpx
import chainlit as cl

API_BASE = "http://localhost:8000/api/v1"


@cl.on_chat_start
async def on_start() -> None:
    cl.user_session.set("session_id", None)
    cl.user_session.set("mode", "qa")

    await cl.Message(
        content=(
            "**PhFlow — FDA Regulatory Intelligence**\n\n"
            "Ask questions about Title 21 CFR (Food and Drugs) regulations.\n\n"
            "**Modes** (type to switch):\n"
            "- `/qa` — Q&A with citations (default)\n"
            "- `/compliance` — Compliance analysis\n"
            "- `/research` — Research summary\n"
        )
    ).send()


@cl.on_message
async def on_message(message: cl.Message) -> None:
    content = message.content.strip()

    # Mode switching commands
    if content.startswith("/"):
        cmd = content.lstrip("/").lower()
        if cmd in ("qa", "compliance", "research"):
            cl.user_session.set("mode", cmd)
            await cl.Message(content=f"Mode switched to **{cmd}**.").send()
            return

    session_id = cl.user_session.get("session_id")
    mode = cl.user_session.get("mode", "qa")

    # Build request
    if session_id:
        url = f"{API_BASE}/chat/{session_id}/message"
    else:
        url = f"{API_BASE}/chat"

    payload = {
        "message": content,
        "mode": mode,
        "filters": {},
        "stream": False,
    }

    async with cl.Step(name=f"Searching regulations ({mode} mode)...") as step:
        try:
            async with httpx.AsyncClient(timeout=120) as client:
                resp = await client.post(url, json=payload)
                resp.raise_for_status()
                data = resp.json()
        except httpx.HTTPStatusError as exc:
            await cl.Message(
                content=f"API error {exc.response.status_code}: {exc.response.text}"
            ).send()
            return
        except Exception as exc:
            await cl.Message(content=f"Error: {exc}").send()
            return

        cl.user_session.set("session_id", data.get("session_id"))
        step.output = f"Found {len(data.get('sources', []))} relevant sections."

    # Build source elements
    elements = []
    for i, source in enumerate(data.get("sources", []), 1):
        label = f"[{i}] {source['section_number']} — {source.get('section_title', '')[:60]}"
        elements.append(
            cl.Text(
                name=label,
                content=source.get("full_text", ""),
                display="side",
            )
        )

    answer = data.get("answer", "")
    citations = data.get("citations_footer", "")
    if citations:
        answer += f"\n\n---\n{citations}"

    await cl.Message(content=answer, elements=elements).send()
