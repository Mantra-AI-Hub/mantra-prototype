"""Vector similarity index with optional FAISS acceleration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np

try:
    import faiss  # type: ignore

    _FAISS_AVAILABLE = True
except Exception:  # pragma: no cover - environment dependent
    faiss = None
    _FAISS_AVAILABLE = False


class VectorIndex:
    """In-memory embedding index with FAISS or numpy cosine fallback."""

    def __init__(self, dimension: int):
        if dimension <= 0:
            raise ValueError("dimension must be > 0")

        self.dimension = int(dimension)
        self._track_ids: List[str] = []
        self._vectors = np.empty((0, self.dimension), dtype=np.float32)

        self._use_faiss = _FAISS_AVAILABLE
        self._index = None
        if self._use_faiss:
            # Inner-product over normalized vectors approximates cosine similarity.
            self._index = faiss.IndexFlatIP(self.dimension)

    @staticmethod
    def _normalize(embedding: Sequence[float], dimension: int) -> np.ndarray:
        vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
        if vector.size != dimension:
            raise ValueError(f"Expected embedding of size {dimension}, got {vector.size}")

        norm = float(np.linalg.norm(vector))
        if norm > 0:
            vector = vector / norm
        return vector

    def add(self, track_id: str, embedding: Sequence[float]) -> None:
        """Add a track embedding to the index."""
        if not track_id:
            raise ValueError("track_id must be non-empty")

        vector = self._normalize(embedding, self.dimension)

        self._track_ids.append(str(track_id))
        if self._use_faiss:
            self._index.add(vector.reshape(1, -1))
        else:
            self._vectors = np.vstack([self._vectors, vector.reshape(1, -1)])

    def search(self, query_embedding: Sequence[float], top_k: int = 5) -> List[Tuple[str, float]]:
        """Search nearest embeddings and return (track_id, score) pairs."""
        if top_k <= 0:
            return []
        if not self._track_ids:
            return []

        query = self._normalize(query_embedding, self.dimension)
        k = min(int(top_k), len(self._track_ids))

        if self._use_faiss:
            scores, indices = self._index.search(query.reshape(1, -1), k)
            result: List[Tuple[str, float]] = []
            for idx, score in zip(indices[0].tolist(), scores[0].tolist()):
                if idx < 0:
                    continue
                result.append((self._track_ids[idx], float(score)))
            return result

        scores = np.dot(self._vectors, query)
        ranked = np.argsort(-scores)[:k]
        return [(self._track_ids[int(i)], float(scores[int(i)])) for i in ranked]

    def save(self, path: str) -> None:
        """Persist index contents to disk."""
        base = Path(path)
        metadata = {
            "dimension": self.dimension,
            "track_ids": self._track_ids,
            "backend": "faiss" if self._use_faiss else "numpy",
        }

        if self._use_faiss:
            faiss_path = base.with_suffix(base.suffix + ".faiss")
            meta_path = base.with_suffix(base.suffix + ".meta.json")
            faiss.write_index(self._index, str(faiss_path))
            meta_path.write_text(json.dumps(metadata), encoding="utf-8")
            return

        npz_path = base.with_suffix(base.suffix + ".npz")
        np.savez_compressed(npz_path, vectors=self._vectors, track_ids=np.array(self._track_ids, dtype=object), dimension=self.dimension)

    @classmethod
    def load(cls, path: str) -> "VectorIndex":
        """Load an index from disk."""
        base = Path(path)
        faiss_path = base.with_suffix(base.suffix + ".faiss")
        meta_path = base.with_suffix(base.suffix + ".meta.json")
        npz_path = base.with_suffix(base.suffix + ".npz")

        if faiss_path.exists() and meta_path.exists() and _FAISS_AVAILABLE:
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
            obj = cls(dimension=int(metadata["dimension"]))
            obj._track_ids = [str(value) for value in metadata.get("track_ids", [])]
            obj._use_faiss = True
            obj._index = faiss.read_index(str(faiss_path))
            return obj

        if npz_path.exists():
            payload = np.load(npz_path, allow_pickle=True)
            obj = cls(dimension=int(payload["dimension"]))
            obj._use_faiss = False
            obj._index = None
            obj._vectors = np.asarray(payload["vectors"], dtype=np.float32)
            obj._track_ids = [str(value) for value in payload["track_ids"].tolist()]
            return obj

        raise FileNotFoundError(f"No saved vector index found for base path: {path}")
