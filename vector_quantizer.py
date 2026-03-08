"""Product-quantization style vector compression."""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np


def _kmeans(data: np.ndarray, k: int, iterations: int = 12, seed: int = 23):
    rng = np.random.default_rng(seed)
    n, _ = data.shape
    if n == 0:
        return np.empty((0, data.shape[1]), dtype=np.float32)

    k = max(1, min(int(k), n))
    centroids = data[rng.choice(n, size=k, replace=False)].astype(np.float32, copy=True)

    for _ in range(max(1, int(iterations))):
        dists = np.linalg.norm(data[:, None, :] - centroids[None, :, :], axis=2)
        labels = np.argmin(dists, axis=1)
        updated = centroids.copy()
        for cid in range(k):
            members = data[labels == cid]
            if members.size > 0:
                updated[cid] = members.mean(axis=0)
        if np.allclose(updated, centroids, atol=1e-5):
            centroids = updated
            break
        centroids = updated

    return centroids.astype(np.float32)


class VectorQuantizer:
    def __init__(self, subspaces: int = 4, codebook_size: int = 16):
        self.subspaces = int(max(1, subspaces))
        self.codebook_size = int(max(2, codebook_size))
        self.codebooks: List[np.ndarray] = []
        self._dims: List[Tuple[int, int]] = []

    def _split_dims(self, dim: int) -> List[Tuple[int, int]]:
        bounds = np.linspace(0, dim, num=self.subspaces + 1, dtype=int)
        return [(int(bounds[i]), int(bounds[i + 1])) for i in range(self.subspaces)]

    def train(self, vectors) -> None:
        arr = np.asarray(vectors, dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError("vectors must be 2D")
        if arr.shape[0] == 0:
            self.codebooks = []
            self._dims = []
            return

        self._dims = self._split_dims(arr.shape[1])
        self.codebooks = []
        for start, end in self._dims:
            segment = arr[:, start:end]
            codebook = _kmeans(segment, self.codebook_size)
            self.codebooks.append(codebook)

    def encode(self, vector) -> Tuple[int, ...]:
        if not self.codebooks:
            raise RuntimeError("VectorQuantizer is not trained")
        v = np.asarray(vector, dtype=np.float32).reshape(-1)
        codes: List[int] = []
        for (start, end), codebook in zip(self._dims, self.codebooks):
            segment = v[start:end]
            dists = np.linalg.norm(codebook - segment[None, :], axis=1)
            codes.append(int(np.argmin(dists)))
        return tuple(codes)

    def decode(self, code: Sequence[int]) -> np.ndarray:
        if not self.codebooks:
            raise RuntimeError("VectorQuantizer is not trained")
        if len(code) != len(self.codebooks):
            raise ValueError("Code length mismatch")

        parts = []
        for idx, codebook in zip(code, self.codebooks):
            cid = int(idx)
            cid = max(0, min(cid, codebook.shape[0] - 1))
            parts.append(codebook[cid])
        return np.concatenate(parts).astype(np.float32)
