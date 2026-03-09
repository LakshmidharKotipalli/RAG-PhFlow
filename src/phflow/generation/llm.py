"""
Async client for the llama.cpp HTTP server.

Start the server with:
    llama-server -m /path/to/model.gguf -c 32768 --port 8080

Uses the native POST /completion endpoint (stable across all llama.cpp
versions). Messages are formatted into a ChatML prompt so instruction-tuned
models receive properly structured input.

Streaming uses Server-Sent Events (SSE): each line is "data: {json}" with a
"content" field; the final event has "stop": true.
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

    @staticmethod
    def _messages_to_prompt(messages: list[dict]) -> str:
        """
        Format a chat messages list into a ChatML prompt string.

        ChatML is the standard format for instruction-tuned models
        (Qwen, Llama-3, Mistral, etc.):
            <|im_start|>system
            ...
            <|im_end|>
            <|im_start|>user
            ...
            <|im_end|>
            <|im_start|>assistant
        """
        prompt = ""
        for msg in messages:
            prompt += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
        prompt += "<|im_start|>assistant\n"
        return prompt

    def _payload(self, prompt: str, stream: bool) -> dict:
        return {
            "prompt": prompt,
            "n_predict": self.max_tokens,
            "temperature": self.temperature,
            "top_p": 0.9,
            "n_ctx": self.num_ctx,
            "stream": stream,
            "stop": ["<|im_end|>", "<|im_start|>"],
        }

    async def generate(self, messages: list[dict]) -> str:
        """Generate a complete response (non-streaming)."""
        prompt = self._messages_to_prompt(messages)
        resp = await self._client.post(
            f"{self.base_url}/completion",
            json=self._payload(prompt, stream=False),
        )
        resp.raise_for_status()
        return resp.json().get("content", "")

    async def stream(self, messages: list[dict]) -> AsyncIterator[str]:
        """Stream response tokens via SSE."""
        prompt = self._messages_to_prompt(messages)
        async with self._client.stream(
            "POST",
            f"{self.base_url}/completion",
            json=self._payload(prompt, stream=True),
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    try:
                        data = json.loads(line[6:])
                    except json.JSONDecodeError:
                        continue
                    token = data.get("content", "")
                    if token:
                        yield token
                    if data.get("stop", False):
                        break

    async def health(self) -> bool:
        """Return True if the llama.cpp server is reachable and ready."""
        try:
            resp = await self._client.get(f"{self.base_url}/health", timeout=5)
            return resp.json().get("status") == "ok"
        except Exception:
            return False

    async def close(self) -> None:
        await self._client.aclose()
