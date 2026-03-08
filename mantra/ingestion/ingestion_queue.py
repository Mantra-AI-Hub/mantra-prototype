"""Ingestion queue with retry metadata for resilient ingestion."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, Optional


@dataclass
class IngestionJob:
    task_type: str
    payload: Dict[str, Any]
    attempts: int = 0


class IngestionQueue:
    def __init__(self):
        self._items: Deque[IngestionJob] = deque()

    def enqueue(self, task_type: str, payload: Dict[str, Any]) -> None:
        self._items.append(IngestionJob(task_type=task_type, payload=dict(payload), attempts=0))

    def dequeue(self) -> Optional[IngestionJob]:
        if not self._items:
            return None
        return self._items.popleft()

    def requeue(self, job: IngestionJob) -> None:
        job.attempts += 1
        self._items.append(job)

    def size(self) -> int:
        return len(self._items)
