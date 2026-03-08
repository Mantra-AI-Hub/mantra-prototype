"""Bulk dataset ingestion helpers."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import List, Tuple

from mantra.ingestion.progress_tracker import IngestionProgressTracker
from mantra.ingestion.dataset_scanner import stream_audio_files
from mantra.pipeline.job_queue import JobQueue


def scan_folder(path: str) -> List[str]:
    """Return all WAV file paths found recursively in a folder."""
    root = Path(path)
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"Dataset folder not found: {path}")
    files = list(stream_audio_files(path))
    files.sort()
    return files


def _build_track_id(file_path: str) -> str:
    digest = hashlib.sha1(file_path.encode("utf-8")).hexdigest()[:12]
    stem = Path(file_path).stem.replace(" ", "_")
    return f"{stem}_{digest}"


def enqueue_all_audio_files(
    path: str,
    queue: JobQueue,
    embedding_path: str,
    progress_tracker: IngestionProgressTracker | None = None,
) -> Tuple[int, int]:
    """Enqueue indexing task for each audio file in dataset folder."""
    files = scan_folder(path)

    failed = 0
    queued_track_ids: List[str] = []

    for file_path in files:
        try:
            file_bytes = Path(file_path).read_bytes()
            track_id = _build_track_id(file_path)
            queue.enqueue(
                "index_audio",
                {
                    "track_id": track_id,
                    "filename": Path(file_path).name,
                    "embedding_path": embedding_path,
                    "audio_bytes": file_bytes,
                },
            )
            queued_track_ids.append(track_id)
        except OSError:
            failed += 1

    if progress_tracker is not None:
        progress_tracker.start_batch(total_files=len(files), pending_track_ids=queued_track_ids, failed_files=failed)

    return len(queued_track_ids), failed
