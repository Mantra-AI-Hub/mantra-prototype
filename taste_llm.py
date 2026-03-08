"""LLM-style user taste modeling with deterministic local fallback."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, Iterable, List


class TasteLLM:
    def __init__(self):
        self.user_events: Dict[str, List[Dict[str, object]]] = defaultdict(list)

    def ingest_events(self, user_id: str, events: Iterable[Dict[str, object]]) -> None:
        self.user_events[str(user_id)].extend(dict(e) for e in events)

    def generate_taste_summary(self, user_id: str) -> str:
        events = self.user_events.get(str(user_id), [])
        if not events:
            return "Taste profile is cold-start; no strong signals yet."
        genres = Counter(str(e.get("genre") or "") for e in events if e.get("genre"))
        artists = Counter(str(e.get("artist") or "") for e in events if e.get("artist"))
        top_genres = ", ".join(g for g, _ in genres.most_common(3)) or "mixed genres"
        top_artists = ", ".join(a for a, _ in artists.most_common(3)) or "emerging artists"
        return f"User prefers {top_genres} and frequently returns to {top_artists}."

    def predict_next_genres_artists(self, user_id: str) -> Dict[str, List[str]]:
        events = self.user_events.get(str(user_id), [])
        genres = Counter(str(e.get("genre") or "") for e in events if e.get("genre"))
        artists = Counter(str(e.get("artist") or "") for e in events if e.get("artist"))
        return {
            "genres": [g for g, _ in genres.most_common(5)],
            "artists": [a for a, _ in artists.most_common(5)],
        }

    def explain_recommendations(self, user_id: str, tracks: Iterable[Dict[str, object]]) -> str:
        summary = self.generate_taste_summary(user_id)
        track_ids = [str(t.get("track_id")) for t in tracks][:5]
        return f"{summary} Recommended next: {', '.join(track_ids)}."

