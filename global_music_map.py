"""Regional music taste mapping."""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from typing import Dict, List


class GlobalMusicMap:
    def __init__(self) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self.region_tastes: Dict[str, Dict[str, int]] = defaultdict(dict)
        self.metrics: Dict[str, float | int] = {"updates": 0, "regions": 0}

    def map_regional_music_taste(self, events: List[Dict[str, object]]) -> Dict[str, Dict[str, int]]:
        grouped: Dict[str, List[str]] = defaultdict(list)
        for event in events:
            grouped[str(event.get("region", "global"))].append(str(event.get("genre", "unknown")))
        mapped: Dict[str, Dict[str, int]] = {}
        for region, genres in grouped.items():
            mapped[region] = dict(Counter(genres))
        self.region_tastes = mapped
        self.metrics["updates"] = int(self.metrics["updates"]) + 1
        self.metrics["regions"] = len(mapped)
        return mapped

    def top_genre_by_region(self) -> Dict[str, str]:
        output: Dict[str, str] = {}
        for region, counts in self.region_tastes.items():
            if not counts:
                continue
            output[region] = max(counts.items(), key=lambda x: x[1])[0]
        return output

    def metrics_snapshot(self) -> Dict[str, float | int]:
        return dict(self.metrics)
