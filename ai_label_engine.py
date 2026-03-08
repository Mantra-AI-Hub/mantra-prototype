"""AI label simulator for artist discovery and promotion."""

from __future__ import annotations

from collections import Counter
from typing import Dict, Iterable, List


class AILabelEngine:
    def __init__(self):
        self.artist_scores: Counter = Counter()
        self.fan_growth: Dict[str, int] = {}

    def discover_new_artists(self, artist_metrics: Iterable[Dict[str, object]], top_k: int = 10) -> List[Dict[str, object]]:
        for item in artist_metrics:
            artist = str(item.get("artist") or "")
            if not artist:
                continue
            score = float(item.get("growth", 0.0))
            self.artist_scores[artist] += score
        return [{"artist": a, "score": s} for a, s in self.artist_scores.most_common(max(0, int(top_k)))]

    def generate_promotional_playlists(self, artists: Iterable[str], length: int = 10) -> Dict[str, List[str]]:
        artist_list = [str(a) for a in artists]
        if not artist_list:
            return {"new_talent_showcase": []}
        tracks = []
        per_artist = max(1, int(length // max(1, len(artist_list))))
        for artist in artist_list:
            for idx in range(per_artist):
                tracks.append(f"{artist}_promo_{idx}")
        return {"new_talent_showcase": tracks[: max(0, int(length))]}

    def track_fan_growth(self, artist: str, new_fans: int) -> Dict[str, object]:
        self.fan_growth[str(artist)] = self.fan_growth.get(str(artist), 0) + int(new_fans)
        return {"artist": str(artist), "fans": self.fan_growth[str(artist)]}
