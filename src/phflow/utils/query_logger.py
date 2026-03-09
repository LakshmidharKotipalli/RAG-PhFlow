"""
Per-query pipeline logger.

Creates one timestamped log file per query inside query_logs/ at the
project root. Each file records every step of the RAG pipeline:
  1. Original query
  2. Corrected/rewritten query
  3. Qdrant retrieval candidates (pre-rerank)
  4. Reranked results (post-rerank, with scores)
  5. Assembled LLM prompt
  6. Raw LLM answer
  7. Final parsed answer + declared sources
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


# query_logs/ lives at the project root  (src/phflow/utils/ → 3 parents up)
_LOG_DIR = Path(__file__).parents[3] / "query_logs"


class QueryLogger:
    """Accumulates pipeline steps in memory and flushes to a file at the end."""

    def __init__(self, query: str) -> None:
        _LOG_DIR.mkdir(exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        safe = "".join(c if c.isalnum() or c in " -" else "_" for c in query[:40]).strip()
        safe = safe.replace(" ", "_")
        self._path = _LOG_DIR / f"{ts}__{safe}.log"
        self._lines: list[str] = [
            f"{'=' * 70}",
            f"QUERY LOG  {datetime.now().isoformat()}",
            f"{'=' * 70}",
            "",
        ]

    def log(self, step: str, data: object) -> None:
        """Append a named step and its data to the in-memory buffer."""
        self._lines.append(f"--- {step} ---")
        if isinstance(data, (dict, list)):
            try:
                self._lines.append(json.dumps(data, indent=2, default=str))
            except Exception:
                self._lines.append(repr(data))
        else:
            self._lines.append(str(data))
        self._lines.append("")

    def save(self) -> None:
        """Write the accumulated log to disk. Silently ignores I/O errors."""
        try:
            self._path.write_text("\n".join(self._lines), encoding="utf-8")
        except Exception:
            pass
