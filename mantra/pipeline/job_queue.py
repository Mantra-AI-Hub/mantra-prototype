"""Simple in-memory job queue for background tasks."""

from __future__ import annotations

import threading
import uuid
from collections import deque
from typing import Any, Deque, Dict, Optional


class JobQueue:
    """Thread-safe FIFO job queue."""

    def __init__(self):
        self._items: Deque[Dict[str, Any]] = deque()
        self._lock = threading.Lock()

    def enqueue(self, task_type: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        job = {
            "job_id": uuid.uuid4().hex,
            "task_type": str(task_type),
            "payload": dict(payload),
        }
        with self._lock:
            self._items.append(job)
        return job

    def dequeue(self) -> Optional[Dict[str, Any]]:
        with self._lock:
            if not self._items:
                return None
            return self._items.popleft()

    def size(self) -> int:
        with self._lock:
            return len(self._items)
