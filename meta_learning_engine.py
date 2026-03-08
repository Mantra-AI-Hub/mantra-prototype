"""Meta-learning engine for ranking pipeline selection."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional


class MetaLearningEngine:
    def __init__(self, store_path: str = "data/meta_learning_experiments.json") -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self.store_path = Path(store_path)
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        self.experiments: List[Dict[str, object]] = []
        self.metrics: Dict[str, float | int | str] = {
            "records": 0,
            "learn_calls": 0,
            "best_score": 0.0,
            "best_pipeline": "",
        }
        self._load()

    def _load(self) -> None:
        if not self.store_path.exists():
            return
        try:
            self.experiments = json.loads(self.store_path.read_text(encoding="utf-8"))
            self.metrics["records"] = len(self.experiments)
        except (OSError, json.JSONDecodeError):
            self.logger.warning("Failed to load experiments from %s", self.store_path)
            self.experiments = []

    def _save(self) -> None:
        self.store_path.write_text(json.dumps(self.experiments, indent=2), encoding="utf-8")

    def store_experiment_results(
        self,
        pipeline_name: str,
        metrics: Dict[str, float],
        metadata: Optional[Dict[str, object]] = None,
    ) -> Dict[str, object]:
        quality = float(metrics.get("quality", 0.0))
        latency = float(metrics.get("latency", 1.0))
        engagement = float(metrics.get("engagement", 0.0))
        score = 0.6 * quality + 0.4 * engagement - 0.1 * latency
        row: Dict[str, object] = {
            "pipeline_name": str(pipeline_name),
            "metrics": {k: float(v) for k, v in metrics.items()},
            "metadata": metadata or {},
            "score": float(score),
        }
        self.experiments.append(row)
        self.metrics["records"] = len(self.experiments)
        self._save()
        self.logger.info("Stored experiment for %s score=%.4f", pipeline_name, score)
        return row

    def learn_best_ranking_pipeline(self) -> Optional[Dict[str, object]]:
        self.metrics["learn_calls"] = int(self.metrics["learn_calls"]) + 1
        if not self.experiments:
            return None
        best = max(self.experiments, key=lambda row: float(row.get("score", 0.0)))
        self.metrics["best_score"] = float(best.get("score", 0.0))
        self.metrics["best_pipeline"] = str(best.get("pipeline_name", ""))
        return dict(best)

    def recommend_best_model(self) -> Optional[str]:
        best = self.learn_best_ranking_pipeline()
        return str(best["pipeline_name"]) if best else None

    def metrics_snapshot(self) -> Dict[str, float | int | str]:
        return dict(self.metrics)
