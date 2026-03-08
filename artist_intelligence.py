"""Artist-level intelligence and recommendation utilities."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, List, Tuple


class ArtistIntelligence:
    def __init__(self):
        self.artist_graph: Dict[str, Dict[str, float]] = defaultdict(dict)
        self.user_artist_counts: Dict[str, Counter] = defaultdict(Counter)

    def build_artist_graph(self, track_metadata: List[Dict[str, object]]) -> Dict[str, Dict[str, float]]:
        by_artist: Dict[str, List[Dict[str, object]]] = defaultdict(list)
        for track in track_metadata:
            artist = str(track.get("artist") or "")
            if not artist:
                continue
            by_artist[artist].append(track)

        artists = sorted(by_artist.keys())
        graph: Dict[str, Dict[str, float]] = defaultdict(dict)
        for i, a in enumerate(artists):
            genres_a = {str(t.get("genre") or "") for t in by_artist[a]}
            years_a = {int(t.get("year")) for t in by_artist[a] if t.get("year") is not None}
            for j, b in enumerate(artists):
                if i == j:
                    continue
                genres_b = {str(t.get("genre") or "") for t in by_artist[b]}
                years_b = {int(t.get("year")) for t in by_artist[b] if t.get("year") is not None}
                genre_overlap = len(genres_a.intersection(genres_b))
                year_overlap = 1 if years_a and years_b and min(abs(ya - yb) for ya in years_a for yb in years_b) <= 2 else 0
                score = float(genre_overlap) + float(year_overlap) * 0.5
                if score > 0:
                    graph[a][b] = score
        self.artist_graph = graph
        return graph

    def compute_artist_similarity(self, artist_id: str, top_k: int = 10) -> List[Tuple[str, float]]:
        neighbors = self.artist_graph.get(str(artist_id), {})
        ranked = sorted(neighbors.items(), key=lambda x: x[1], reverse=True)
        return [(artist, float(score)) for artist, score in ranked[: max(0, int(top_k))]]

    def record_user_artist(self, user_id: str, artist_id: str) -> None:
        if artist_id:
            self.user_artist_counts[str(user_id)][str(artist_id)] += 1

    def recommend_artists(self, user_id: str, k: int = 10) -> List[Tuple[str, float]]:
        artists = self.user_artist_counts.get(str(user_id), Counter())
        if not artists:
            return []
        scores: Counter = Counter()
        for artist, weight in artists.items():
            for candidate, sim in self.artist_graph.get(artist, {}).items():
                if candidate == artist:
                    continue
                scores[candidate] += float(weight) * float(sim)
        ranked = scores.most_common(max(0, int(k)))
        return [(artist, float(score)) for artist, score in ranked]


