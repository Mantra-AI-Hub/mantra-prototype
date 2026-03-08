"""Model evolution manager for versioning and weak-model pruning."""

from __future__ import annotations

import logging
from typing import Dict, List

from mantra.model_registry import ModelRegistry


class ModelEvolutionManager:
    def __init__(self, registry: ModelRegistry | None = None) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self.registry = registry or ModelRegistry(root_dir="models")
        self.scores: Dict[str, float] = {}
        self.metrics: Dict[str, float | int] = {"versions": 0, "pruned": 0}

    def version_model(self, model_name: str, version: str, path: str, score: float = 0.0) -> Dict[str, str]:
        record = self.registry.register(model_name=model_name, version=version, path=path)
        key = f"{model_name}:{version}"
        self.scores[key] = float(score)
        self.metrics["versions"] = int(self.metrics["versions"]) + 1
        return record

    def prune_weak_models(self, threshold: float = 0.5) -> List[str]:
        removed = [key for key, score in self.scores.items() if float(score) < float(threshold)]
        for key in removed:
            del self.scores[key]
        if removed:
            self.logger.info("Pruned %d weak model versions", len(removed))
        self.metrics["pruned"] = int(self.metrics["pruned"]) + len(removed)
        return removed

    def metrics_snapshot(self) -> Dict[str, float | int]:
        return {**self.metrics, "tracked_models": len(self.scores)}
