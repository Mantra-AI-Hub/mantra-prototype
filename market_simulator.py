"""Streaming market economy simulation."""

from __future__ import annotations

import logging
from typing import Dict


class MarketSimulator:
    def __init__(self) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self.metrics: Dict[str, float | int] = {"runs": 0}

    def simulate_streaming_platform_economy(
        self, streams: int, payout_per_stream: float = 0.003, platform_fee: float = 0.3
    ) -> Dict[str, float]:
        gross = float(max(0, int(streams)) * max(0.0, float(payout_per_stream)))
        fee = gross * max(0.0, min(1.0, float(platform_fee)))
        artist_revenue = gross - fee
        self.metrics["runs"] = int(self.metrics["runs"]) + 1
        self.logger.info("Economy simulation streams=%d artist_revenue=%.4f", streams, artist_revenue)
        return {"gross_revenue": gross, "platform_fee": fee, "artist_revenue": artist_revenue}

    def metrics_snapshot(self) -> Dict[str, float | int]:
        return dict(self.metrics)
