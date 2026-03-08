"""Playlist recommendation and exposure optimization."""

from __future__ import annotations

import logging
from typing import Dict, List


class PromotionOptimizer:
    def __init__(self) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self.metrics: Dict[str, float | int] = {"recommendations": 0, "optimizations": 0}

    def recommend_playlists(self, artist_genre: str, playlists: List[Dict[str, object]], top_k: int = 5) -> List[Dict[str, object]]:
        ranked = sorted(
            playlists,
            key=lambda item: float(str(artist_genre).lower() in [str(g).lower() for g in item.get("genres", [])]),
            reverse=True,
        )
        self.metrics["recommendations"] = int(self.metrics["recommendations"]) + 1
        return ranked[: max(1, int(top_k))]

    def optimize_exposure(self, current_reach: float, budget: float) -> Dict[str, float]:
        gain = float(max(0.0, budget) * 0.08)
        projected = float(max(0.0, current_reach) + gain)
        self.metrics["optimizations"] = int(self.metrics["optimizations"]) + 1
        self.logger.info("Exposure optimized reach %.2f -> %.2f", current_reach, projected)
        return {"current_reach": float(current_reach), "projected_reach": projected, "gain": gain}

    def metrics_snapshot(self) -> Dict[str, float | int]:
        return dict(self.metrics)
