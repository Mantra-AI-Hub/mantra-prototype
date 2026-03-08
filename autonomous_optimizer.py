"""Autonomous optimization loop for recommendation parameter tuning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass
class OptimizerState:
    ranking_weight: float = 0.8
    exploration: float = 0.1
    cluster_size: int = 4
    iterations: int = 0


class AutonomousOptimizer:
    def __init__(self):
        self.state = OptimizerState()
        self.last_metrics: Dict[str, float] = {}

    def monitor_metrics(self, metrics: Dict[str, float]) -> Dict[str, float]:
        self.last_metrics = {k: float(v) for k, v in metrics.items()}
        return self.last_metrics

    def optimize(self) -> Dict[str, object]:
        ctr = float(self.last_metrics.get("ctr", 0.0))
        latency = float(self.last_metrics.get("latency", 0.0))
        if ctr < 0.2:
            self.state.exploration = min(0.5, self.state.exploration + 0.02)
        else:
            self.state.exploration = max(0.01, self.state.exploration - 0.01)
        if latency > 0.2:
            self.state.ranking_weight = max(0.5, self.state.ranking_weight - 0.02)
        else:
            self.state.ranking_weight = min(0.95, self.state.ranking_weight + 0.01)
        self.state.cluster_size = max(2, min(16, self.state.cluster_size + (1 if ctr > 0.3 else 0)))
        self.state.iterations += 1
        return {
            "ranking_weight": self.state.ranking_weight,
            "exploration": self.state.exploration,
            "cluster_size": self.state.cluster_size,
            "iterations": self.state.iterations,
        }

