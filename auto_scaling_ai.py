"""Automatic scaling controller for AI services."""

from __future__ import annotations

import logging
from typing import Dict


class AutoScalingAI:
    def __init__(self, min_replicas: int = 1, max_replicas: int = 10) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self.min_replicas = max(1, int(min_replicas))
        self.max_replicas = max(self.min_replicas, int(max_replicas))
        self.current_replicas = self.min_replicas
        self.metrics: Dict[str, float | int] = {"scale_events": 0}

    def dynamically_scale_models(self, load: float, latency: float) -> int:
        target = self.current_replicas
        if float(load) > 0.75 or float(latency) > 0.4:
            target += 1
        elif float(load) < 0.3 and float(latency) < 0.2:
            target -= 1
        target = max(self.min_replicas, min(self.max_replicas, target))
        if target != self.current_replicas:
            self.current_replicas = target
            self.metrics["scale_events"] = int(self.metrics["scale_events"]) + 1
            self.logger.info("Scaled replicas to %d", target)
        return self.current_replicas

    def metrics_snapshot(self) -> Dict[str, float | int]:
        return {**self.metrics, "replicas": self.current_replicas}
