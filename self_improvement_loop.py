"""Continuous self-improvement loop for metrics, retraining, and deployment."""

from __future__ import annotations

import logging
from typing import Dict

from mantra.ai_music_supervisor import AIMusicSupervisor
from mantra.model_evolution_manager import ModelEvolutionManager


class SelfImprovementLoop:
    def __init__(
        self,
        supervisor: AIMusicSupervisor | None = None,
        evolution_manager: ModelEvolutionManager | None = None,
    ) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self.supervisor = supervisor or AIMusicSupervisor()
        self.evolution_manager = evolution_manager or ModelEvolutionManager()
        self.metrics: Dict[str, float | int] = {"iterations": 0}

    def collect_metrics(self) -> Dict[str, object]:
        return self.supervisor.status()

    def retrain_models(self, performance_score: float) -> Dict[str, object]:
        return self.supervisor.run_supervision_cycle(performance_score=performance_score)

    def redeploy_best_models(self, model_name: str = "ranking", version: str = "auto", path: str = "models/ranking.pkl") -> Dict[str, str]:
        return self.evolution_manager.version_model(model_name=model_name, version=version, path=path, score=1.0)

    def run_iteration(self, performance_score: float = 0.6) -> Dict[str, object]:
        status = self.retrain_models(performance_score=performance_score)
        self.metrics["iterations"] = int(self.metrics["iterations"]) + 1
        self.logger.info("Self-improvement iteration %d", self.metrics["iterations"])
        return {"status": status, "metrics": dict(self.metrics)}
