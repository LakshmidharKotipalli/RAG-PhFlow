"""
Async client for the llama.cpp HTTP server.

Start the server with:
    llama-server -m /path/to/model.gguf -c 32768 --port 8080

The server exposes:
  POST /v1/chat/completions  — OpenAI-compatible chat endpoint (used here)
  GET  /health               — liveness probe

Using the chat completions endpoint ensures the model's chat template is
applied automatically, so instruction-tuned models correctly generate answers
rather than continuing the prompt as raw text completion.

Streaming uses Server-Sent Events (SSE): each event is a JSON line with a
"choices[0].delta.content" field; the stream ends with "data: [DONE]".
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
        self.max_tokens = max_tokens if max_tokens is not None else settings.llama_cpp_max_tokens
        self._client = httpx.AsyncClient(timeout=timeout)

    def _chat_payload(self, messages: list[dict], stream: bool) -> dict:
        return {
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": 0.9,
            "stream": stream,
        }

    async def generate(self, messages: list[dict]) -> str:
        """Generate a complete response (non-streaming)."""
        resp = await self._client.post(
            f"{self.base_url}/v1/chat/completions",
            json=self._chat_payload(messages, stream=False),
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    async def stream(self, messages: list[dict]) -> AsyncIterator[str]:
        """Stream response tokens via SSE (OpenAI-compatible format)."""
        async with self._client.stream(
            "POST",
            f"{self.base_url}/v1/chat/completions",
            json=self._chat_payload(messages, stream=True),
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    raw = line[6:]
                    if raw.strip() == "[DONE]":
                        break
                    try:
                        data = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    token = data.get("choices", [{}])[0].get("delta", {}).get("content", "")
                    if token:
                        yield token

    async def health(self) -> bool:
        """Return True if the llama.cpp server is reachable and ready."""
        try:
            resp = await self._client.get(f"{self.base_url}/health", timeout=5)
            body = resp.json()
            return body.get("status") == "ok"
        except Exception:
            return False

    async def close(self) -> None:
        await self._client.aclose()
