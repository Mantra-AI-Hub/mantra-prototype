"""Background worker for processing queue tasks."""

from __future__ import annotations

import threading
import time
from typing import Dict, Optional

from mantra.database.track_store import TrackStore
from mantra.pipeline.job_queue import JobQueue
from mantra.pipeline.tasks import index_audio_task
from mantra.vector_index.faiss_index import VectorIndex


class BackgroundWorker:
    """Worker loop consuming queued tasks and persisting artifacts."""

    def __init__(
        self,
        queue: JobQueue,
        vector_index: VectorIndex,
        track_store: TrackStore,
        fingerprint_db: Dict[str, list],
        vector_index_path: str,
        poll_interval: float = 0.1,
    ):
        self.queue = queue
        self.vector_index = vector_index
        self.track_store = track_store
        self.fingerprint_db = fingerprint_db
        self.vector_index_path = vector_index_path
        self.poll_interval = float(poll_interval)
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def process_one(self) -> bool:
        job = self.queue.dequeue()
        if job is None:
            return False

        task_type = job.get("task_type")
        payload = dict(job.get("payload", {}))

        if task_type == "index_audio":
            result = index_audio_task(payload)
            track_id = str(result["track_id"])

            self.vector_index.add(track_id, result["embedding"])
            self.vector_index.save(self.vector_index_path)

            self.fingerprint_db[track_id] = result["fingerprint"]
            self.track_store.add_track(
                {
                    "track_id": track_id,
                    "filename": result["filename"],
                    "duration": result["duration"],
                    "embedding_path": result["embedding_path"],
                    "fingerprint_hash_count": result["fingerprint_hash_count"],
                    "created_at": result["created_at"],
                    "artist": result.get("artist", ""),
                    "album": result.get("album", ""),
                    "genre": result.get("genre", ""),
                    "tags": result.get("tags", []),
                    "year": result.get("year"),
                }
            )
            self.track_store.upsert_fingerprint(track_id, result["fingerprint"])
            return True

        return False

    def run(self) -> None:
        while not self._stop_event.is_set():
            processed = self.process_one()
            if not processed:
                time.sleep(self.poll_interval)

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self.run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None
