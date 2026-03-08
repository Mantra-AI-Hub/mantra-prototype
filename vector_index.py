"""Large-scale vector index interface backed by FAISS/numpy."""

from __future__ import annotations

from typing import List, Sequence, Tuple

from mantra.vector_index.faiss_index import VectorIndex as _BaseVectorIndex


class VectorIndex:
    """Compatibility interface: add(vector, track_id), search(vector, k), save/load."""

    def __init__(self, dimension: int):
        self._index = _BaseVectorIndex(dimension=dimension)

    @property
    def dimension(self) -> int:
        return int(self._index.dimension)

    def add(self, vector: Sequence[float], track_id: str) -> None:
        self._index.add(track_id=str(track_id), embedding=vector)

    def search(self, vector: Sequence[float], k: int) -> List[Tuple[str, float]]:
        return self._index.search(query_embedding=vector, top_k=k)

    def save(self, path: str) -> None:
        self._index.save(path)

    @classmethod
    def load(cls, path: str) -> "VectorIndex":
        base = _BaseVectorIndex.load(path)
        obj = cls(dimension=base.dimension)
        obj._index = base
        return obj

