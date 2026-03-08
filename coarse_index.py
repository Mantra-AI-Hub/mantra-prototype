"""Two-stage coarse index for candidate cluster selection."""

from __future__ import annotations

from typing import List

import numpy as np


def _kmeans(vectors: np.ndarray, k: int, iterations: int = 15, seed: int = 11):
    rng = np.random.default_rng(seed)
    n, _ = vectors.shape
    if n == 0:
        return np.empty((0, vectors.shape[1]), dtype=np.float32), np.empty((0,), dtype=np.int32)

    k = max(1, min(int(k), n))
    init_idx = rng.choice(n, size=k, replace=False)
    centroids = vectors[init_idx].astype(np.float32, copy=True)

    labels = np.zeros(n, dtype=np.int32)
    for _ in range(max(1, int(iterations))):
        dists = np.linalg.norm(vectors[:, None, :] - centroids[None, :, :], axis=2)
        labels = np.argmin(dists, axis=1).astype(np.int32)

        updated = centroids.copy()
        for cid in range(k):
            members = vectors[labels == cid]
            if members.size == 0:
                updated[cid] = vectors[rng.integers(0, n)]
            else:
                updated[cid] = members.mean(axis=0)
        if np.allclose(updated, centroids, atol=1e-5):
            centroids = updated
            break
        centroids = updated

    return centroids.astype(np.float32), labels


class CoarseIndex:
    def __init__(self, clusters: int = 32):
        self.clusters = int(max(1, clusters))
        self.centroids = np.empty((0, 0), dtype=np.float32)
        self.assignments = np.empty((0,), dtype=np.int32)

    def train(self, vectors) -> None:
        arr = np.asarray(vectors, dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError("vectors must be 2D")
        if arr.shape[0] == 0:
            self.centroids = np.empty((0, arr.shape[1]), dtype=np.float32)
            self.assignments = np.empty((0,), dtype=np.int32)
            return
        self.centroids, self.assignments = _kmeans(arr, self.clusters)

    def assign(self, vector) -> int:
        if self.centroids.size == 0:
            return -1
        query = np.asarray(vector, dtype=np.float32).reshape(-1)
        dists = np.linalg.norm(self.centroids - query[None, :], axis=1)
        return int(np.argmin(dists))

    def get_candidate_clusters(self, vector, n: int = 5) -> List[int]:
        if self.centroids.size == 0:
            return []
        query = np.asarray(vector, dtype=np.float32).reshape(-1)
        dists = np.linalg.norm(self.centroids - query[None, :], axis=1)
        order = np.argsort(dists)
        top = max(1, min(int(n), order.size))
        return [int(v) for v in order[:top].tolist()]
