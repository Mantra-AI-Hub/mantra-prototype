"""Graph-based recommendation utilities."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple

try:
    import networkx as nx  # type: ignore
except Exception:  # pragma: no cover
    nx = None


class GraphRecommender:
    def __init__(self):
        self._fallback_user_tracks: Dict[str, Dict[str, float]] = defaultdict(dict)
        self._fallback_track_users: Dict[str, Dict[str, float]] = defaultdict(dict)
        self._graph = None

    def build_user_track_graph(self, interactions: List[Dict[str, object]]) -> None:
        if nx is not None:
            graph = nx.Graph()
            for item in interactions:
                user = f"u:{item.get('user_id')}"
                track = f"t:{item.get('track_id')}"
                weight = float(item.get("weight", 1.0))
                graph.add_edge(user, track, weight=weight)
            self._graph = graph
        else:
            self._fallback_user_tracks.clear()
            self._fallback_track_users.clear()
            for item in interactions:
                user = str(item.get("user_id"))
                track = str(item.get("track_id"))
                weight = float(item.get("weight", 1.0))
                self._fallback_user_tracks[user][track] = self._fallback_user_tracks[user].get(track, 0.0) + weight
                self._fallback_track_users[track][user] = self._fallback_track_users[track].get(user, 0.0) + weight

    def compute_personalized_pagerank(self, user_id: str) -> Dict[str, float]:
        if nx is not None and self._graph is not None:
            source = f"u:{user_id}"
            if source not in self._graph:
                return {}
            personalization = {node: 0.0 for node in self._graph.nodes}
            personalization[source] = 1.0
            scores = nx.pagerank(self._graph, personalization=personalization, alpha=0.85)
            return {k[2:]: float(v) for k, v in scores.items() if str(k).startswith("t:")}

        # Fallback: weighted two-hop user-track-user-track expansion.
        seen_tracks = self._fallback_user_tracks.get(str(user_id), {})
        scores: Dict[str, float] = defaultdict(float)
        for track, w_ut in seen_tracks.items():
            for other_user, w_tu in self._fallback_track_users.get(track, {}).items():
                for candidate_track, w_uct in self._fallback_user_tracks.get(other_user, {}).items():
                    if candidate_track in seen_tracks:
                        continue
                    scores[candidate_track] += float(w_ut * w_tu * w_uct)
        return dict(scores)

    def recommend_from_graph(self, user_id: str, k: int) -> List[Tuple[str, float]]:
        scores = self.compute_personalized_pagerank(user_id)
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [(tid, float(score)) for tid, score in ranked[: max(0, int(k))]]
