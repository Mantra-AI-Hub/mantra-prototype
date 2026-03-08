"""Dynamic recommendation engine orchestration for streaming use-cases."""

from __future__ import annotations

from typing import Dict


class StreamingRecommendationOrchestrator:
    def __init__(self):
        self.engine_health = {
            "vector": 1.0,
            "graph": 0.9,
            "gnn": 0.85,
            "bandit": 0.8,
            "transformer": 0.88,
        }

    def choose_best_engine(self, context: Dict[str, object] | None = None) -> str:
        ctx = context or {}
        if ctx.get("cold_start"):
            return "bandit"
        return max(self.engine_health.items(), key=lambda x: x[1])[0]

    def update_engine_health(self, engine: str, score: float) -> None:
        if engine in self.engine_health:
            self.engine_health[engine] = float(max(0.0, min(1.0, score)))

    def status(self) -> Dict[str, object]:
        return {"engine_health": dict(self.engine_health), "selected": self.choose_best_engine({})}

