"""
Async client for the llama.cpp HTTP server.

Start the server with:
    llama-server -m /path/to/model.gguf -c 32768 --port 8080

Uses the native POST /completion endpoint with a plain-text labeled-section
prompt that works with any instruction-tuned model regardless of its specific
chat template (Llama-3, Mistral, Qwen, etc.).

Streaming uses Server-Sent Events (SSE): each line is "data: {json}" with a
"content" field; the final event has "stop": true.
"""

from __future__ import annotations

import json
import re
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

    # EOS / role tokens sent to llama.cpp as stop sequences.
    # Do NOT include prompt section labels here — if the model echoes one
    # as its first token, generation stops immediately and _clean() would
    # strip it, leaving an empty answer.
    _EOS_TOKENS = [
        "<|im_end|>", "<|im_start|>", "<|ir_end|>",
        "</s>", "<|endoftext|>", "<|eot_id|>",
    ]

    # Section labels that should be stripped from the start/end of the
    # generated text if the model leaks them, but must NOT be stop sequences.
    _LABEL_TOKENS = [
        "INSTRUCTIONS:", "CONTEXT:", "QUESTION:", "CONVERSATION HISTORY:", "Answer:",
    ]

    def _payload(self, prompt: str, stream: bool) -> dict:
        return {
            "prompt": prompt,
            "n_predict": self.max_tokens,
            "temperature": self.temperature,
            "top_p": 0.9,
            "n_ctx": self.num_ctx,
            "stream": stream,
            "stop": self._EOS_TOKENS,
            # Penalise repeated token sequences to prevent the model from
            # looping on the SOURCES_USED line or echoing the prompt back.
            "repeat_penalty": 1.1,
            "repeat_last_n": 256,
            "frequency_penalty": 0.1,
        }

    @classmethod
    def _clean(cls, text: str) -> str:
        """Strip leaked EOS tokens, section-label tokens, and reasoning blocks."""
        result = text.strip()

        # Remove <think>...</think> reasoning blocks emitted by reasoning models
        # (e.g. Qwen3, DeepSeek-R1).  The blocks may span multiple lines and
        # may be empty.  Strip all occurrences, not just the first.
        result = re.sub(r"<think>.*?</think>", "", result, flags=re.DOTALL).strip()

        # Strip EOS / section-label tokens from the start and end.
        changed = True
        while changed:
            changed = False
            for token in (*cls._EOS_TOKENS, *cls._LABEL_TOKENS):
                if result.startswith(token):
                    result = result[len(token):].strip()
                    changed = True
                if result.endswith(token):
                    result = result[:-len(token)].strip()
                    changed = True
        return result

    async def correct_query(self, prompt: str) -> str:
        """
        Run a short correction pass on the user query.
        Uses a tight token budget since the output is a single sentence.
        Falls back to the original if the corrected result is empty.
        """
        payload = self._payload(prompt, stream=False)
        payload["n_predict"] = 128   # one sentence is enough
        resp = await self._client.post(
            f"{self.base_url}/completion",
            json=payload,
        )
        resp.raise_for_status()
        corrected = self._clean(resp.json().get("content", "")).strip()
        return corrected or ""   # caller decides fallback

    async def generate(self, prompt: str) -> str:
        """Generate a complete response (non-streaming)."""
        resp = await self._client.post(
            f"{self.base_url}/completion",
            json=self._payload(prompt, stream=False),
        )
        resp.raise_for_status()
        return self._clean(resp.json().get("content", ""))

    async def stream(self, prompt: str) -> AsyncIterator[str]:
        """Stream response tokens via SSE."""
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
                    token = self._clean(data.get("content", ""))
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
