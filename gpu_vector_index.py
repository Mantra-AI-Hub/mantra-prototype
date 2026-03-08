"""GPU-accelerated ANN search with CPU fallback."""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

from mantra.vector_index import VectorIndex

try:
    import faiss  # type: ignore
except Exception:  # pragma: no cover
    faiss = None


class GPUVectorIndex:
    """Load CPU index, optionally move FAISS index to GPU, and serve ANN search."""

    def __init__(self):
        self.cpu_index: VectorIndex | None = None
        self.gpu_index = None
        self.track_ids: List[str] = []
        self.gpu_enabled = False

    def load_cpu_index(self, path: str) -> None:
        self.cpu_index = VectorIndex.load(path)

        # Best-effort extraction of track IDs from wrapped index internals.
        base = getattr(self.cpu_index, "_index", None)
        ids = getattr(base, "_track_ids", None)
        self.track_ids = list(ids) if isinstance(ids, list) else []

        self.gpu_index = None
        self.gpu_enabled = False

    def to_gpu(self, device_id: int = 0) -> bool:
        if self.cpu_index is None:
            raise RuntimeError("CPU index must be loaded before GPU transfer")
        if faiss is None:
            self.gpu_enabled = False
            return False

        try:
            base = getattr(self.cpu_index, "_index", None)
            faiss_cpu_index = getattr(base, "_index", None)
            if faiss_cpu_index is None or not hasattr(faiss, "StandardGpuResources"):
                self.gpu_enabled = False
                return False

            resources = faiss.StandardGpuResources()
            self.gpu_index = faiss.index_cpu_to_gpu(resources, int(device_id), faiss_cpu_index)
            self.gpu_enabled = True
            return True
        except Exception:
            self.gpu_index = None
            self.gpu_enabled = False
            return False

    def _cpu_search(self, vector: Sequence[float], k: int) -> List[Tuple[str, float]]:
        if self.cpu_index is None:
            return []
        return self.cpu_index.search(vector, k)

    def search(self, vector: Sequence[float], k: int) -> List[Tuple[str, float]]:
        if k <= 0:
            return []
        if not self.gpu_enabled or self.gpu_index is None:
            return self._cpu_search(vector, k)

        query = np.asarray(vector, dtype=np.float32).reshape(1, -1)
        scores, indices = self.gpu_index.search(query, int(k))
        results: List[Tuple[str, float]] = []

        for idx, score in zip(indices[0].tolist(), scores[0].tolist()):
            if idx < 0:
                continue
            track_id = self.track_ids[idx] if idx < len(self.track_ids) else str(idx)
            results.append((track_id, float(score)))

        if not results:
            return self._cpu_search(vector, k)
        return results

    def add(self, vector: Sequence[float], track_id: str) -> None:
        if self.cpu_index is None:
            raise RuntimeError("CPU index must be loaded before add")

        self.cpu_index.add(vector, track_id)
        self.track_ids.append(str(track_id))

        if self.gpu_enabled and self.gpu_index is not None:
            arr = np.asarray(vector, dtype=np.float32).reshape(1, -1)
            self.gpu_index.add(arr)
