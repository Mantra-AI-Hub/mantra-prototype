"""Artist growth and breakout prediction."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, Iterable, List


class ArtistGrowthPredictor:
    def __init__(self):
        self.artist_events: Dict[str, List[Dict[str, object]]] = defaultdict(list)

    def ingest_events(self, events: Iterable[Dict[str, object]]) -> None:
        for e in events:
            artist = str(e.get("artist") or "")
            if artist:
                self.artist_events[artist].append(dict(e))

    def predict_artist_popularity_growth(self, artist: str) -> float:
        events = self.artist_events.get(str(artist), [])
        plays = sum(1 for e in events if str(e.get("event")) in {"play", "click", "like"})
        likes = sum(1 for e in events if str(e.get("event")) == "like")
        return float(plays * 0.7 + likes * 1.3)

    def predict_breakout_artists(self, top_k: int = 10) -> List[Dict[str, object]]:
        scores = []
        for artist in self.artist_events:
            scores.append({"artist": artist, "growth": self.predict_artist_popularity_growth(artist)})
        scores.sort(key=lambda x: x["growth"], reverse=True)
        return scores[: max(0, int(top_k))]

    def analyze_audience_expansion(self, artist: str) -> Dict[str, object]:
        users = Counter(str(e.get("user_id") or "") for e in self.artist_events.get(str(artist), []) if e.get("user_id"))
        return {"artist": artist, "unique_listeners": len(users), "top_listeners": [u for u, _ in users.most_common(5)]}

