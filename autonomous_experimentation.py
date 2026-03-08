"""Autonomous A/B experimentation for ranking pipelines."""

from __future__ import annotations

import logging
import uuid
from typing import Dict, List, Optional


class AutonomousExperimentation:
    def __init__(self) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self.experiments: Dict[str, Dict[str, object]] = {}
        self.deployed_pipeline: str = ""
        self.metrics: Dict[str, float | int | str] = {
            "launched": 0,
            "evaluated": 0,
            "deployments": 0,
            "active_experiments": 0,
        }

    def launch_ab_experiment(self, pipeline_a: str, pipeline_b: str, split: float = 0.5) -> Dict[str, object]:
        exp_id = f"exp_{uuid.uuid4().hex[:10]}"
        record = {
            "experiment_id": exp_id,
            "pipeline_a": str(pipeline_a),
            "pipeline_b": str(pipeline_b),
            "split": float(max(0.05, min(0.95, split))),
            "winner": None,
            "status": "running",
        }
        self.experiments[exp_id] = record
        self.metrics["launched"] = int(self.metrics["launched"]) + 1
        self.metrics["active_experiments"] = len([e for e in self.experiments.values() if e["status"] == "running"])
        self.logger.info("Launched experiment %s", exp_id)
        return dict(record)

    def compare_ranking_pipelines(self, scores_a: List[float], scores_b: List[float]) -> Dict[str, object]:
        mean_a = float(sum(scores_a) / max(1, len(scores_a)))
        mean_b = float(sum(scores_b) / max(1, len(scores_b)))
        winner = "A" if mean_a >= mean_b else "B"
        return {"mean_a": mean_a, "mean_b": mean_b, "winner": winner}

    def evaluate_experiment(self, experiment_id: str, scores_a: List[float], scores_b: List[float]) -> Dict[str, object]:
        exp = self.experiments.get(str(experiment_id))
        if not exp:
            raise ValueError(f"Unknown experiment_id: {experiment_id}")
        outcome = self.compare_ranking_pipelines(scores_a, scores_b)
        exp["winner"] = outcome["winner"]
        exp["status"] = "evaluated"
        self.metrics["evaluated"] = int(self.metrics["evaluated"]) + 1
        self.metrics["active_experiments"] = len([e for e in self.experiments.values() if e["status"] == "running"])
        return {"experiment_id": experiment_id, **outcome}

    def deploy_best_performer(self, experiment_id: str) -> Optional[str]:
        exp = self.experiments.get(str(experiment_id))
        if not exp or exp.get("winner") not in {"A", "B"}:
            return None
        winner = str(exp["winner"])
        pipeline_name = str(exp["pipeline_a"] if winner == "A" else exp["pipeline_b"])
        self.deployed_pipeline = pipeline_name
        exp["status"] = "deployed"
        self.metrics["deployments"] = int(self.metrics["deployments"]) + 1
        self.logger.info("Deployed pipeline %s from %s", pipeline_name, experiment_id)
        return pipeline_name

    def metrics_snapshot(self) -> Dict[str, float | int | str]:
        return {**self.metrics, "deployed_pipeline": self.deployed_pipeline}
