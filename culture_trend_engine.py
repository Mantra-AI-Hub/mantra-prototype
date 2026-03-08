"""Cultural trend detection and genre evolution tracking."""

from __future__ import annotations

import logging
from collections import Counter
from typing import Dict, List


class CultureTrendEngine:
    def __init__(self) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self.genre_history: List[str] = []
        self.metrics: Dict[str, float | int] = {"shift_runs": 0, "evolution_runs": 0}

    def detect_cultural_music_shifts(self, events: List[Dict[str, object]], top_k: int = 5) -> List[Dict[str, object]]:
        genres = [str(item.get("genre", "unknown")) for item in events]
        self.genre_history.extend(genres)
        counts = Counter(genres)
        self.metrics["shift_runs"] = int(self.metrics["shift_runs"]) + 1
        return [{"genre": genre, "count": int(count)} for genre, count in counts.most_common(max(1, int(top_k)))]

    def track_genre_evolution(self) -> Dict[str, float]:
        total = max(1, len(self.genre_history))
        counts = Counter(self.genre_history)
        self.metrics["evolution_runs"] = int(self.metrics["evolution_runs"]) + 1
        evolution = {genre: float(count / total) for genre, count in counts.items()}
        self.logger.info("Tracked evolution over %d genre events", total)
        return evolution

    def metrics_snapshot(self) -> Dict[str, float | int]:
        return dict(self.metrics)
