"""
FAISS-based vector store with JSON metadata sidecar.

All dense vectors live in a FAISS IndexFlatIP (inner-product, equivalent to
cosine for L2-normalised vectors).  Metadata payloads are stored in a
parallel list serialised as JSON.

GPU acceleration:  The store attempts to use ``faiss-gpu`` when a CUDA device
is available.  If the GPU transfer fails for any reason the store falls back
to the CPU index and logs a warning — the search results are identical either
way; only throughput differs.

Directory layout:
    <faiss_index_dir>/
        dense.index          – FAISS binary index (always saved as CPU)
        metadata.json        – list[dict] aligned by row id
        bm25_sparse.json     – sparse BM25 data (indices + values per doc)
"""

from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ── lazy FAISS import ────────────────────────────────────────
_faiss = None


def _get_faiss():
    global _faiss
    if _faiss is None:
        import faiss
        _faiss = faiss
    return _faiss


def _try_gpu_index(cpu_index):
    """
    Attempt to move a CPU FAISS index to GPU 0.

    Returns (gpu_index, True) on success or (cpu_index, False) on failure.
    A warning is logged whenever the GPU path is unavailable.
    """
    faiss = _get_faiss()

    # 1. Check if faiss was built with GPU support
    if not hasattr(faiss, "index_cpu_to_gpu"):
        logger.warning(
            "FAISS GPU: faiss-gpu not installed — falling back to CPU. "
            "Install faiss-gpu for GPU-accelerated search."
        )
        return cpu_index, False

    try:
        res = faiss.StandardGpuResources()
        gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
        logger.info("FAISS GPU: index moved to GPU 0 successfully")
        return gpu_index, True
    except Exception as exc:
        logger.warning(
            "FAISS GPU: failed to move index to GPU — falling back to CPU. "
            "Reason: %s",
            exc,
        )
        return cpu_index, False


class FaissStore:
    """Thin wrapper around a FAISS flat index + metadata sidecar."""

    def __init__(
        self,
        index_dir: str | Path,
        dim: int = 768,
        use_gpu: bool = True,
    ) -> None:
        self.index_dir = Path(index_dir)
        self.dim = dim
        self._use_gpu = use_gpu
        self._on_gpu = False

        self._index = None                # faiss.IndexFlatIP (CPU or GPU)
        self._cpu_index = None            # always kept for saving
        self._metadata: list[dict] = []   # one dict per vector row
        self._sparse_data: list[dict] = []  # BM25 sparse vectors per row
        self._id_map: dict[str, int] = {}   # point_id (str) -> row index

        if self._index_path.exists():
            self.load()
        else:
            faiss = _get_faiss()
            cpu_idx = faiss.IndexFlatIP(dim)
            self._cpu_index = cpu_idx
            if use_gpu:
                self._index, self._on_gpu = _try_gpu_index(cpu_idx)
            else:
                self._index = cpu_idx
            logger.info(
                "Created new FAISS index (dim=%d, device=%s)",
                dim,
                "GPU" if self._on_gpu else "CPU",
            )

    # ── file paths ──────────────────────────────────────────────

    @property
    def _index_path(self) -> Path:
        return self.index_dir / "dense.index"

    @property
    def _meta_path(self) -> Path:
        return self.index_dir / "metadata.json"

    @property
    def _sparse_path(self) -> Path:
        return self.index_dir / "bm25_sparse.json"

    # ── persistence ─────────────────────────────────────────────

    def save(self) -> None:
        """Persist index + metadata to disk.  Index is always saved as CPU."""
        self.index_dir.mkdir(parents=True, exist_ok=True)
        faiss = _get_faiss()

        # Always save the CPU copy
        save_idx = self._cpu_index if self._cpu_index is not None else self._index
        if self._on_gpu and hasattr(faiss, "index_gpu_to_cpu"):
            save_idx = faiss.index_gpu_to_cpu(self._index)

        faiss.write_index(save_idx, str(self._index_path))
        self._meta_path.write_text(
            json.dumps(self._metadata, ensure_ascii=False), encoding="utf-8"
        )
        self._sparse_path.write_text(
            json.dumps(self._sparse_data, ensure_ascii=False), encoding="utf-8"
        )
        self._rebuild_id_map()
        logger.info(
            "FAISS store saved: %d vectors → %s", save_idx.ntotal, self.index_dir
        )

    def load(self) -> None:
        """Load index + metadata from disk, optionally moving to GPU."""
        faiss = _get_faiss()
        cpu_idx = faiss.read_index(str(self._index_path))
        self._cpu_index = cpu_idx

        if self._use_gpu:
            self._index, self._on_gpu = _try_gpu_index(cpu_idx)
        else:
            self._index = cpu_idx

        self._metadata = json.loads(self._meta_path.read_text(encoding="utf-8"))
        if self._sparse_path.exists():
            self._sparse_data = json.loads(
                self._sparse_path.read_text(encoding="utf-8")
            )
        else:
            self._sparse_data = []
        self._rebuild_id_map()
        logger.info(
            "FAISS store loaded: %d vectors from %s (device=%s)",
            self._index.ntotal,
            self.index_dir,
            "GPU" if self._on_gpu else "CPU",
        )

    def _rebuild_id_map(self) -> None:
        self._id_map = {
            m.get("_point_id", ""): i for i, m in enumerate(self._metadata)
        }

    # ── queries ─────────────────────────────────────────────────

    @property
    def total_points(self) -> int:
        return self._index.ntotal if self._index else 0

    def exists(self) -> bool:
        return self._index_path.exists()

    def search(
        self,
        query_vector: list[float] | np.ndarray,
        limit: int = 10,
        filter_fn: Any = None,
    ) -> list[dict]:
        """
        Search the FAISS index and return top-k results.

        Args:
            query_vector: Dense query vector (768-dim, L2-normalised).
            limit: Max results to return.
            filter_fn: Optional callable(payload_dict) -> bool.
                       If provided, only results where filter_fn returns True
                       are included.  We over-fetch to compensate for filtering.

        Returns:
            List of dicts with keys: id, score, payload.
        """
        if self._index is None or self._index.ntotal == 0:
            return []

        vec = np.array([query_vector], dtype=np.float32)

        # Over-fetch when filtering to ensure we get enough results
        search_k = limit * 5 if filter_fn else limit
        search_k = min(search_k, self._index.ntotal)

        scores, indices = self._index.search(vec, search_k)

        results: list[dict] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            payload = self._metadata[idx]
            if filter_fn and not filter_fn(payload):
                continue
            results.append({
                "id": payload.get("_point_id", str(idx)),
                "score": float(score),
                "payload": payload,
            })
            if len(results) >= limit:
                break

        return results

    def scroll(
        self,
        filter_fn: Any = None,
        limit: int = 100,
    ) -> list[dict]:
        """
        Iterate over stored metadata (no vector search).

        Args:
            filter_fn: Optional callable(payload_dict) -> bool.
            limit: Max results.

        Returns:
            List of dicts with keys: id, payload.
        """
        results: list[dict] = []
        for i, payload in enumerate(self._metadata):
            if filter_fn and not filter_fn(payload):
                continue
            results.append({
                "id": payload.get("_point_id", str(i)),
                "payload": payload,
            })
            if len(results) >= limit:
                break
        return results

    def scroll_by_node(self, section_node: str, limit: int = 100) -> list[dict]:
        """Fetch all chunks belonging to a section node, ordered by chunk_index."""
        results = self.scroll(
            filter_fn=lambda p: p.get("section_node") == section_node,
            limit=limit,
        )
        results.sort(key=lambda r: r["payload"].get("chunk_index", 0))
        return results

    def scroll_by_field(self, field: str, value: str, limit: int = 100) -> list[dict]:
        """Fetch all chunks matching a metadata field value, ordered by chunk_index."""
        results = self.scroll(
            filter_fn=lambda p: p.get(field) == value,
            limit=limit,
        )
        results.sort(key=lambda r: r["payload"].get("chunk_index", 0))
        return results

    # ── mutations ───────────────────────────────────────────────

    def add(
        self,
        dense_vectors: list[list[float]] | np.ndarray,
        payloads: list[dict],
        sparse_data: list[dict] | None = None,
    ) -> None:
        """
        Add vectors and their metadata to the store (in-memory only).
        Call save() afterwards to persist.

        Args:
            dense_vectors: List of 768-dim float vectors.
            payloads: Parallel list of metadata dicts.
            sparse_data: Optional parallel list of BM25 sparse dicts
                         with keys 'indices' and 'values'.
        """
        vecs = np.array(dense_vectors, dtype=np.float32)
        assert vecs.shape[1] == self.dim, f"Expected dim={self.dim}, got {vecs.shape[1]}"

        # Assign unique point IDs
        for p in payloads:
            if "_point_id" not in p:
                p["_point_id"] = str(uuid.uuid4())

        # Add to CPU index first (always kept in sync for saving)
        if self._cpu_index is not None:
            self._cpu_index.add(vecs)

        # Add to the active index (GPU or CPU)
        if self._on_gpu and self._index is not self._cpu_index:
            self._index.add(vecs)
        elif self._cpu_index is None:
            self._index.add(vecs)

        self._metadata.extend(payloads)
        if sparse_data:
            self._sparse_data.extend(sparse_data)
        else:
            self._sparse_data.extend([{}] * len(payloads))

        # Update id map
        start = len(self._id_map)
        for i, p in enumerate(payloads):
            self._id_map[p["_point_id"]] = start + i

    def reset(self) -> None:
        """Drop all data and recreate empty index."""
        faiss = _get_faiss()
        cpu_idx = faiss.IndexFlatIP(self.dim)
        self._cpu_index = cpu_idx
        if self._use_gpu:
            self._index, self._on_gpu = _try_gpu_index(cpu_idx)
        else:
            self._index = cpu_idx
            self._on_gpu = False
        self._metadata = []
        self._sparse_data = []
        self._id_map = {}
        # Delete files if they exist
        for p in (self._index_path, self._meta_path, self._sparse_path):
            if p.exists():
                p.unlink()
        logger.info(
            "FAISS store reset (dim=%d, device=%s)",
            self.dim,
            "GPU" if self._on_gpu else "CPU",
        )

    def get_sparse_data(self) -> list[dict]:
        """Return the sparse BM25 data for all stored vectors."""
        return self._sparse_data


def get_faiss_store(
    index_dir: str | None = None,
    dim: int = 768,
    use_gpu: bool = True,
) -> FaissStore:
    """Factory: create or load a FaissStore from settings."""
    from phflow.settings import settings
    path = index_dir or settings.faiss_index_dir
    return FaissStore(index_dir=path, dim=dim, use_gpu=use_gpu)
