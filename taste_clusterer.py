"""User taste clustering with sklearn KMeans and fallback."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, Iterable, List

import numpy as np

try:
    from sklearn.cluster import KMeans  # type: ignore
except Exception:  # pragma: no cover
    KMeans = None


class TasteClusterer:
    def __init__(self, n_clusters: int = 4):
        self.n_clusters = max(1, int(n_clusters))
        self.user_embeddings: Dict[str, np.ndarray] = {}
        self.user_clusters: Dict[str, int] = {}
        self.cluster_track_counts: Dict[int, Counter] = defaultdict(Counter)
        self._kmeans = None
        self._user_order: List[str] = []

    def cluster_users(self, user_embeddings: Dict[str, Iterable[float]]) -> Dict[str, int]:
        vectors: List[np.ndarray] = []
        users: List[str] = []
        for user_id, embedding in user_embeddings.items():
            vec = np.asarray(list(embedding), dtype=np.float32).reshape(-1)
            if vec.size == 0:
                continue
            users.append(str(user_id))
            vectors.append(vec)
            self.user_embeddings[str(user_id)] = vec
        if not users:
            return {}

        max_dim = max(int(v.size) for v in vectors)
        matrix = np.zeros((len(vectors), max_dim), dtype=np.float32)
        for i, vec in enumerate(vectors):
            matrix[i, : vec.size] = vec

        if KMeans is not None and len(users) >= self.n_clusters:
            self._kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init="auto")
            labels = self._kmeans.fit_predict(matrix)
            self.user_clusters = {users[i]: int(labels[i]) for i in range(len(users))}
        else:
            self._kmeans = None
            self._user_order = list(users)
            self.user_clusters = {uid: int(abs(hash(uid)) % self.n_clusters) for uid in users}
        return dict(self.user_clusters)

    def assign_user_cluster(self, user_id: str) -> int:
        uid = str(user_id)
        if uid in self.user_clusters:
            return int(self.user_clusters[uid])
        vec = self.user_embeddings.get(uid)
        if vec is None:
            cluster_id = int(abs(hash(uid)) % self.n_clusters)
            self.user_clusters[uid] = cluster_id
            return cluster_id
        if self._kmeans is not None:
            shaped = vec.reshape(1, -1)
            cluster_id = int(self._kmeans.predict(shaped)[0])
        else:
            cluster_id = int(abs(hash(uid)) % self.n_clusters)
        self.user_clusters[uid] = cluster_id
        return cluster_id

    def record_interaction(self, user_id: str, track_id: str) -> None:
        cluster_id = self.assign_user_cluster(user_id)
        self.cluster_track_counts[cluster_id][str(track_id)] += 1

    def recommend_from_cluster(self, cluster_id: int, k: int = 10) -> List[str]:
        counts = self.cluster_track_counts.get(int(cluster_id), Counter())
        return [track_id for track_id, _ in counts.most_common(max(0, int(k)))]


