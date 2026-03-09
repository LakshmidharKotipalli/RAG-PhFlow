"""
Async client for the llama.cpp HTTP server.

Start the server with:
    llama-server -m /path/to/model.gguf -c 32768 --port 8080

The server exposes:
  POST /completion  — native completion endpoint (used here)
  GET  /health      — liveness probe

Streaming uses Server-Sent Events (SSE): each event is a JSON line with
a "content" field; the final event has "stop": true.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

import httpx

from phflow.config.settings import settings


class LlamaCppClient:
    def __init__(
        self,
        base_url: str | None = None,
        temperature: float | None = None,
        num_ctx: int | None = None,
        max_tokens: int | None = None,
        timeout: int = 300,
    ) -> None:
        if base_url is not None:
            raw = base_url
            if not raw.startswith(("http://", "https://")):
                raw = "http://" + raw
            self.base_url = raw.rstrip("/")
        else:
            self.base_url = settings.llama_cpp_base_url
        self.temperature = temperature if temperature is not None else settings.llama_cpp_temperature
        self.num_ctx = num_ctx if num_ctx is not None else settings.llama_cpp_num_ctx
        self.max_tokens = max_tokens if max_tokens is not None else settings.llama_cpp_max_tokens
        self._client = httpx.AsyncClient(timeout=timeout)

    def _payload(self, prompt: str, stream: bool) -> dict:
        return {
            "prompt": prompt,
            "n_predict": self.max_tokens,
            "temperature": self.temperature,
            "top_p": 0.9,
            "n_ctx": self.num_ctx,
            "stream": stream,
            "stop": ["</s>", "<|im_end|>", "QUESTION:", "REGULATORY CONTEXT:"],
        }

    async def generate(self, prompt: str) -> str:
        """Generate a complete response (non-streaming)."""
        resp = await self._client.post(
            f"{self.base_url}/completion",
            json=self._payload(prompt, stream=False),
        )
        resp.raise_for_status()
        return resp.json().get("content", "")

    async def stream(self, prompt: str) -> AsyncIterator[str]:
        """Stream response tokens via SSE."""
        async with self._client.stream(
            "POST",
            f"{self.base_url}/completion",
            json=self._payload(prompt, stream=True),
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                # llama.cpp SSE format: "data: {...}"
                if line.startswith("data: "):
                    data = json.loads(line[6:])
                    token = data.get("content", "")
                    if token:
                        yield token
                    if data.get("stop", False):
                        break

    async def health(self) -> bool:
        """Return True if the llama.cpp server is reachable and ready."""
        try:
            resp = await self._client.get(f"{self.base_url}/health", timeout=5)
            body = resp.json()
            # llama.cpp /health returns {"status": "ok"} when model is loaded
            return body.get("status") == "ok"
        except Exception:
            return False

    async def close(self) -> None:
        await self._client.aclose()
