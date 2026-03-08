"""Central supervisor coordinating recommendation, generation, and experiments."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Dict, List

from mantra.autonomous_experimentation import AutonomousExperimentation
from mantra.intelligence.music_foundation_model import MusicFoundationModel
from mantra.intelligence.music_genome_engine import MusicGenomeEngine
from mantra.intelligence.music_genome_store import MusicGenomeStore
from mantra.self_evolving_recommender import SelfEvolvingRecommender


class AIMusicSupervisor:
    def __init__(
        self,
        recommender: SelfEvolvingRecommender | None = None,
        experimentation: AutonomousExperimentation | None = None,
        foundation_model: MusicFoundationModel | None = None,
        genome_engine: MusicGenomeEngine | None = None,
        genome_store: MusicGenomeStore | None = None,
    ) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self.recommender = recommender or SelfEvolvingRecommender()
        self.experimentation = experimentation or AutonomousExperimentation()
        self.foundation_model = foundation_model or MusicFoundationModel()
        self.genome_engine = genome_engine or MusicGenomeEngine()
        self.genome_store = genome_store or MusicGenomeStore(genome_engine=self.genome_engine)
        self.metrics: Dict[str, float | int] = {"cycles": 0}
        self.events: List[Dict[str, object]] = []

    def run_supervision_cycle(self, performance_score: float) -> Dict[str, object]:
        self.recommender.monitor_model_performance(performance_score)
        retrained = self.recommender.auto_retrain_models()
        self.metrics["cycles"] = int(self.metrics["cycles"]) + 1
        status = {
            "recommender": self.recommender.metrics_snapshot(),
            "experimentation": self.experimentation.metrics_snapshot(),
            "foundation_model": {"backend": self.foundation_model.backend, "embedding_dim": self.foundation_model.embedding_dim},
            "genome_engine": {"fields": len(self.genome_engine.FIELDS)},
            "genome_store": {"stored_genomes": len(self.genome_store.genomes)},
            "retrained": retrained,
        }
        self.logger.info("Supervisor cycle completed retrained=%s", retrained)
        return status

    def status(self) -> Dict[str, object]:
        return {
            "recommender": self.recommender.metrics_snapshot(),
            "experimentation": self.experimentation.metrics_snapshot(),
            "foundation_model": {"backend": self.foundation_model.backend, "embedding_dim": self.foundation_model.embedding_dim},
            "genome_engine": {"fields": len(self.genome_engine.FIELDS)},
            "genome_store": {"stored_genomes": len(self.genome_store.genomes)},
            "metrics": dict(self.metrics),
            "events": list(self.events),
        }

    def log_event(self, name: str, payload: Dict[str, object] | None = None) -> None:
        entry = {
            "event": name,
            "payload": payload or {},
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self.events.append(entry)
        self.logger.info("Supervisor event %s captured", name)
