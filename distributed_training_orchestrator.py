"""Orchestrator for distributed model training jobs."""

from __future__ import annotations

import logging
import uuid
from typing import Dict


class DistributedTrainingOrchestrator:
    def __init__(self) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self.jobs: Dict[str, Dict[str, object]] = {}
        self.metrics: Dict[str, float | int] = {"submitted": 0, "completed": 0}

    def submit_training_job(self, model_name: str, shards: int = 1) -> Dict[str, object]:
        job_id = f"job_{uuid.uuid4().hex[:10]}"
        record = {"job_id": job_id, "model_name": str(model_name), "shards": max(1, int(shards)), "status": "queued"}
        self.jobs[job_id] = record
        self.metrics["submitted"] = int(self.metrics["submitted"]) + 1
        return dict(record)

    def mark_job_running(self, job_id: str) -> None:
        if job_id in self.jobs:
            self.jobs[job_id]["status"] = "running"

    def complete_job(self, job_id: str) -> None:
        if job_id in self.jobs:
            self.jobs[job_id]["status"] = "completed"
            self.metrics["completed"] = int(self.metrics["completed"]) + 1
            self.logger.info("Completed training job %s", job_id)

    def status(self) -> Dict[str, object]:
        return {"jobs": list(self.jobs.values()), "metrics": dict(self.metrics)}
