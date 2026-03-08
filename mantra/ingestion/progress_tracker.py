"""Track progress for dataset ingestion jobs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Set

from mantra.database.track_store import TrackStore


@dataclass
class IngestionProgressTracker:
    """Maintains counters for current ingestion batch."""

    total_files: int = 0
    processed_files: int = 0
    failed_files: int = 0
    pending_track_ids: Set[str] = field(default_factory=set)

    def reset(self) -> None:
        self.total_files = 0
        self.processed_files = 0
        self.failed_files = 0
        self.pending_track_ids.clear()

    def start_batch(self, total_files: int, pending_track_ids: Iterable[str], failed_files: int = 0) -> None:
        self.total_files = int(total_files)
        self.processed_files = 0
        self.failed_files = int(failed_files)
        self.pending_track_ids = {str(value) for value in pending_track_ids}

    def refresh(self, track_store: TrackStore) -> None:
        resolved = []
        for track_id in self.pending_track_ids:
            metadata = track_store.get_track(track_id)
            if metadata and int(metadata.get("fingerprint_hash_count", 0)) > 0:
                resolved.append(track_id)

        for track_id in resolved:
            self.pending_track_ids.discard(track_id)
            self.processed_files += 1

    def snapshot(self, track_store: TrackStore | None = None) -> Dict[str, int]:
        if track_store is not None:
            self.refresh(track_store)

        return {
            "total_files": int(self.total_files),
            "processed_files": int(self.processed_files),
            "failed_files": int(self.failed_files),
            "queued_files": int(len(self.pending_track_ids)),
        }
