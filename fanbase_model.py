"""Fanbase growth and engagement simulation."""

from __future__ import annotations

import logging
from typing import Dict


class FanbaseModel:
    def __init__(self) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self.fanbase: Dict[str, float] = {}
        self.metrics: Dict[str, float | int] = {"growth_runs": 0, "engagement_updates": 0}

    def simulate_fan_growth(self, artist_id: str, growth_rate: float, periods: int = 4) -> float:
        base = float(self.fanbase.get(str(artist_id), 100.0))
        fans = base * ((1.0 + max(-0.99, float(growth_rate))) ** max(1, int(periods)))
        self.fanbase[str(artist_id)] = float(fans)
        self.metrics["growth_runs"] = int(self.metrics["growth_runs"]) + 1
        return float(fans)

    def engagement_modeling(self, artist_id: str, content_quality: float, release_frequency: float) -> float:
        engagement = max(0.0, min(1.0, 0.7 * float(content_quality) + 0.3 * float(release_frequency)))
        self.metrics["engagement_updates"] = int(self.metrics["engagement_updates"]) + 1
        self.logger.info("Updated engagement score for %s to %.4f", artist_id, engagement)
        return float(engagement)

    def metrics_snapshot(self) -> Dict[str, float | int]:
        return dict(self.metrics)
