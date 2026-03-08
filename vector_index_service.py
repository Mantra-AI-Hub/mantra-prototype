"""Production vector index service with optional FAISS support."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from mantra.vector_index import VectorIndex


class VectorIndexService:
    def __init__(self, index_path: str = "data/vector_index_service"):
        self.index_path = index_path
        self.index: VectorIndex | None = None
        self.dimension: int | None = None
        self.track_vectors: Dict[str, np.ndarray] = {}

    def build_index(self, dimension: int) -> None:
        self.dimension = int(dimension)
        self.index = VectorIndex(dimension=self.dimension)

    def add_vector(self, track_id: str, vector: Sequence[float]) -> None:
        vec = np.asarray(vector, dtype=np.float32).reshape(-1)
        if self.index is None:
            self.build_index(vec.size)
        if vec.size != int(self.dimension):
            raise ValueError(f"Vector dimension mismatch: expected {self.dimension}, got {vec.size}")

        self.index.add(vec, str(track_id))
        self.track_vectors[str(track_id)] = vec

    def search(self, vector: Sequence[float], k: int) -> List[Tuple[str, float]]:
        if self.index is None:
            return []
        return self.index.search(vector, int(k))

    def save_index(self) -> None:
        if self.index is None:
            return
        Path(self.index_path).parent.mkdir(parents=True, exist_ok=True)
        self.index.save(self.index_path)

    def load_index(self) -> None:
        self.index = VectorIndex.load(self.index_path)
        self.dimension = self.index.dimension


_default_service = VectorIndexService()


def build_index(dimension: int) -> None:
    _default_service.build_index(dimension)


def add_vector(track_id: str, vector: Sequence[float]) -> None:
    _default_service.add_vector(track_id, vector)


def search(vector: Sequence[float], k: int):
    return _default_service.search(vector, k)


def save_index() -> None:
    _default_service.save_index()


def load_index() -> None:
    _default_service.load_index()
