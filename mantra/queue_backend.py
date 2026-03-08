"""Pluggable queue backend for distributed worker architecture."""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, Optional


@dataclass
class QueueJob:
    job_id: str
    payload: Dict[str, Any]


class InMemoryQueueBackend:
    def __init__(self):
        self._items: Deque[QueueJob] = deque()
        self._counter = 0

    def enqueue(self, job: Dict[str, Any]) -> QueueJob:
        self._counter += 1
        queue_job = QueueJob(job_id=f"mem-{self._counter}", payload=dict(job))
        self._items.append(queue_job)
        return queue_job

    def dequeue(self) -> Optional[QueueJob]:
        if not self._items:
            return None
        return self._items.popleft()

    def ack(self, job: QueueJob) -> None:
        return None

    def depth(self) -> int:
        return len(self._items)


class RedisQueueBackend:
    """Optional Redis-backed queue backend."""

    def __init__(self, redis_url: str = "redis://localhost:6379/0", key: str = "mantra:jobs"):
        try:
            import redis  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("redis package is required for RedisQueueBackend") from exc

        self._redis = redis.from_url(redis_url)
        self.key = key

    def enqueue(self, job: Dict[str, Any]) -> QueueJob:
        payload = json.dumps(job)
        job_id = self._redis.incr(f"{self.key}:id")
        self._redis.rpush(self.key, payload)
        return QueueJob(job_id=f"redis-{job_id}", payload=dict(job))

    def dequeue(self) -> Optional[QueueJob]:
        value = self._redis.lpop(self.key)
        if value is None:
            return None
        payload = json.loads(value)
        return QueueJob(job_id="redis-pop", payload=payload)

    def ack(self, job: QueueJob) -> None:
        return None

    def depth(self) -> int:
        return int(self._redis.llen(self.key))
