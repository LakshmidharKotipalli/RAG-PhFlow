"""
Per-query pipeline logger.

Writes one timestamped JSON file per query inside query_logs/ at the project
root. Each file records every step of the RAG pipeline:
  1. Original query
  2. Retrieval query (after optional rewrite)
  3. Auto-extracted filters (if any)
  4. Qdrant retrieval candidates (pre-rerank)
  5. Reranked results (post-rerank, with scores)
  6. Sections passed to LLM
  7. Assembled prompt
  8. Raw LLM answer
  9. Final answer + declared sources

Multi-line text fields (assembled prompt, raw LLM answer, final answer) are
stored as arrays of lines so the JSON file is readable in any text editor.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


# query_logs/ lives at the project root  (src/phflow/utils/ → 3 parents up)
_LOG_DIR = Path(__file__).parents[3] / "query_logs"

# Fields that contain long multi-line text — stored as line arrays in JSON
# so they render readably instead of one escaped wall of text.
_MULTILINE_FIELDS = {"ASSEMBLED PROMPT", "RAW LLM ANSWER", "FINAL ANSWER"}


class QueryLogger:
    """Accumulates pipeline steps in memory and flushes to a JSON file at the end."""

    def __init__(self, query: str) -> None:
        _LOG_DIR.mkdir(exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        safe = "".join(c if c.isalnum() or c in " -" else "_" for c in query[:40]).strip()
        safe = safe.replace(" ", "_")
        self._path = _LOG_DIR / f"{ts}__{safe}.json"
        self._data: dict[str, object] = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
        }

    def log(self, step: str, data: object) -> None:
        """Record a named pipeline step."""
        if step in _MULTILINE_FIELDS and isinstance(data, str):
            # Split into lines so the JSON file stays human-readable
            self._data[step] = data.splitlines()
        else:
            self._data[step] = data

    def save(self) -> None:
        """Write accumulated pipeline data to a JSON file. Silently ignores I/O errors."""
        try:
            self._path.write_text(
                json.dumps(self._data, indent=2, default=str), encoding="utf-8"
            )
        except Exception:
            pass

    # Kept for call-site compatibility — delegates to save()
    def save_json(self) -> None:
        self.save()
