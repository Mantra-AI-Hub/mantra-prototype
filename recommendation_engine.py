"""Recommendation engine using similarity graph and diversity filtering."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np

from mantra.intelligence.music_foundation_model import MusicFoundationModel
from mantra.intelligence.music_genome_store import MusicGenomeStore


class RecommendationEngine:
    def __init__(
        self,
        track_store,
        genome_store: MusicGenomeStore | None = None,
        foundation_model: MusicFoundationModel | None = None,
    ):
        self.track_store = track_store
        self.graph: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
        self.genome_store = genome_store
        self.foundation_model = foundation_model or MusicFoundationModel()

    def build_similarity_graph(self) -> Dict[str, List[Tuple[str, float]]]:
        tracks = self.track_store.list_tracks()
        by_genre: Dict[str, List[dict]] = defaultdict(list)
        for track in tracks:
            by_genre[str(track.get("genre") or "")].append(track)

        graph: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
        for genre, items in by_genre.items():
            for i, a in enumerate(items):
                for j, b in enumerate(items):
                    if i == j:
                        continue
                    score = 1.0
                    if a.get("artist") and b.get("artist") and a.get("artist") != b.get("artist"):
                        score = 0.85
                    graph[str(a["track_id"])].append((str(b["track_id"]), float(score)))

        for track_id, edges in graph.items():
            edges.sort(key=lambda item: item[1], reverse=True)
            graph[track_id] = edges

        self.graph = graph
        return graph

    def recommend(self, track_id: str, k: int) -> List[Tuple[str, float]]:
        if not self.graph:
            self.build_similarity_graph()
        base = list(self.graph.get(str(track_id), []))
        if not base:
            return []

        # Simple graph expansion for second-hop candidates.
        second_hop: Dict[str, float] = defaultdict(float)
        for neighbor_id, score in base[: max(1, k * 2)]:
            second_hop[neighbor_id] = max(second_hop[neighbor_id], score)
            for hop2_id, hop2_score in self.graph.get(neighbor_id, [])[:k]:
                second_hop[hop2_id] = max(second_hop[hop2_id], score * hop2_score * 0.9)

        # Diversity filtering: limit repeats from same artist.
        ranked = sorted(second_hop.items(), key=lambda item: item[1], reverse=True)
        artist_seen = set()
        recommendations: List[Tuple[str, float]] = []
        seed_features = self._seed_features(track_id)
        for candidate_id, score in ranked:
            if candidate_id == track_id:
                continue
            metadata = self.track_store.get_track(candidate_id) or {}
            artist = str(metadata.get("artist") or "")
            if artist and artist in artist_seen:
                continue
            if artist:
                artist_seen.add(artist)
            adjusted = float(score + self._genome_similarity(track_id, candidate_id) + self._foundation_similarity(seed_features, metadata))
            recommendations.append((candidate_id, adjusted))
            if len(recommendations) >= max(0, int(k)):
                break

        return recommendations

    def _genome_similarity(self, seed_track_id: str, candidate_track_id: str) -> float:
        if self.genome_store is None:
            return 0.0
        seed = self.genome_store.get_genome(seed_track_id)
        candidate = self.genome_store.get_genome(candidate_track_id)
        if not seed or not candidate:
            return 0.0
        return 0.1 * float(self.genome_store.genome_engine.similarity_score(seed, candidate))

    @staticmethod
    def _seed_features(track_id: str) -> Dict[str, object]:
        return {"track_id": track_id}

    def _foundation_similarity(self, seed_features: Dict[str, object], metadata: Dict[str, object]) -> float:
        seed_vec = self.foundation_model.embed_features(
            [float(hash(str(seed_features.get("track_id", ""))) % 100) / 100.0]
        )
        cand_vec = self.foundation_model.embed_features(
            [
                float(hash(str(metadata.get("genre", ""))) % 100) / 100.0,
                float(hash(str(metadata.get("artist", ""))) % 100) / 100.0,
            ]
        )
        sim = float(np.dot(seed_vec, cand_vec) / max(1e-6, float(np.linalg.norm(seed_vec) * np.linalg.norm(cand_vec))))
        return 0.05 * max(0.0, sim)
