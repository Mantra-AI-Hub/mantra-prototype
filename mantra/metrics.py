"""Metrics collection and Prometheus formatting."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict


@dataclass
class MetricsRegistry:
    ingestion_rate: float = 0.0
    embedding_latency: float = 0.0
    search_latency: float = 0.0
    cache_hit_rate: float = 0.0
    queue_depth: int = 0
    worker_failures: int = 0
    _ingested_items: int = 0
    _ingestion_started: float = 0.0

    def start_ingestion_timer(self) -> None:
        self._ingestion_started = time.perf_counter()

    def record_ingested(self, count: int = 1) -> None:
        self._ingested_items += int(count)
        elapsed = max(1e-6, time.perf_counter() - self._ingestion_started) if self._ingestion_started else 1.0
        self.ingestion_rate = float(self._ingested_items / elapsed)

    def update_cache_hit_rate(self, hits: int, misses: int) -> None:
        total = hits + misses
        self.cache_hit_rate = float(hits / total) if total > 0 else 0.0

    def as_dict(self) -> Dict[str, float | int]:
        return {
            "ingestion_rate": float(self.ingestion_rate),
            "embedding_latency": float(self.embedding_latency),
            "search_latency": float(self.search_latency),
            "cache_hit_rate": float(self.cache_hit_rate),
            "queue_depth": int(self.queue_depth),
            "worker_failures": int(self.worker_failures),
        }

    def prometheus(self) -> str:
        values = self.as_dict()
        lines = []
        for key, value in values.items():
            lines.append(f"# TYPE mantra_{key} gauge")
            lines.append(f"mantra_{key} {value}")
        return "\n".join(lines) + "\n"


metrics_registry = MetricsRegistry()
