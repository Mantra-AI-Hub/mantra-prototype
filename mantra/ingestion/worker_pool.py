"""Parallel worker pool for ingestion processing."""

from __future__ import annotations

import threading
import time
from typing import Callable, Optional

from mantra.ingestion.ingestion_queue import IngestionJob, IngestionQueue


class WorkerPool:
    """Threaded pool that processes ingestion queue jobs with retry support."""

    def __init__(
        self,
        queue: IngestionQueue,
        job_handler: Callable[[IngestionJob], bool],
        workers: int = 4,
        worker_id: str = "worker-0",
        concurrency: int = 1,
        max_batch_size: int = 1,
        max_retries: int = 2,
        poll_interval: float = 0.05,
    ):
        self.queue = queue
        self.job_handler = job_handler
        self.workers = max(1, int(workers))
        self.worker_id = str(worker_id)
        self.concurrency = max(1, int(concurrency))
        self.max_batch_size = max(1, int(max_batch_size))
        self.max_retries = max(0, int(max_retries))
        self.poll_interval = float(poll_interval)
        self._threads: list[threading.Thread] = []
        self._stop = threading.Event()

    def _dequeue_batch(self) -> list[IngestionJob]:
        batch: list[IngestionJob] = []
        for _ in range(self.max_batch_size):
            job = self.queue.dequeue()
            if job is None:
                break
            batch.append(job)
        return batch

    def _run(self, thread_index: int) -> None:
        while not self._stop.is_set():
            batch = self._dequeue_batch()
            if not batch:
                time.sleep(self.poll_interval)
                continue

            for job in batch:
                try:
                    ok = bool(self.job_handler(job))
                except Exception:
                    ok = False

                if not ok and job.attempts < self.max_retries:
                    self.queue.requeue(job)

    def start(self) -> None:
        if self._threads:
            return
        self._stop.clear()
        total_threads = max(1, self.workers * self.concurrency)
        self._threads = [
            threading.Thread(target=self._run, args=(idx,), daemon=True, name=f"{self.worker_id}-{idx}")
            for idx in range(total_threads)
        ]
        for thread in self._threads:
            thread.start()

    def stop(self, timeout: Optional[float] = 2.0) -> None:
        self._stop.set()
        for thread in self._threads:
            thread.join(timeout=timeout)
        self._threads = []
