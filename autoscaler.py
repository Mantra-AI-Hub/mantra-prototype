"""Autoscaling controller for worker infrastructure."""

from __future__ import annotations

from typing import Dict


class AutoScaler:
    def __init__(self, min_workers: int = 1, max_workers: int = 16):
        self.min_workers = int(max(1, min_workers))
        self.max_workers = int(max(self.min_workers, max_workers))
        self.current_workers = self.min_workers

    def monitor_metrics(self, metrics: Dict[str, float | int]) -> Dict[str, object]:
        queue_depth = float(metrics.get("queue_depth", 0))
        ingestion_rate = float(metrics.get("ingestion_rate", 0))
        worker_failures = float(metrics.get("worker_failures", 0))

        if worker_failures > 0:
            return self.scale_down(reason="worker_failures")
        if queue_depth > max(10.0, ingestion_rate * 2.0):
            return self.scale_up(reason="queue_depth")
        if queue_depth < 2.0:
            return self.scale_down(reason="low_load")
        return {"action": "hold", "workers": self.current_workers}

    def scale_up(self, reason: str = "") -> Dict[str, object]:
        self.current_workers = min(self.max_workers, self.current_workers + 1)
        return {"action": "scale_up", "workers": self.current_workers, "reason": reason}

    def scale_down(self, reason: str = "") -> Dict[str, object]:
        self.current_workers = max(self.min_workers, self.current_workers - 1)
        return {"action": "scale_down", "workers": self.current_workers, "reason": reason}
