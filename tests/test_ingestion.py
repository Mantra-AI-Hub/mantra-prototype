import io
import shutil
import wave
from pathlib import Path
from uuid import uuid4

import numpy as np

import mantra.interfaces.api.api_server as api_server
from mantra.database.track_store import TrackStore
from mantra.ingestion.dataset_ingestor import enqueue_all_audio_files, scan_folder
from mantra.ingestion.progress_tracker import IngestionProgressTracker
from mantra.pipeline.job_queue import JobQueue
from mantra.pipeline.worker import BackgroundWorker
from mantra.vector_engine.embedding_builder import EMBEDDING_SIZE
from mantra.vector_index.faiss_index import VectorIndex


def _make_wav_bytes(freq: float, duration: float = 1.0, sample_rate: int = 22050) -> bytes:
    t = np.linspace(0.0, duration, int(sample_rate * duration), endpoint=False, dtype=np.float32)
    signal = 0.5 * np.sin(2.0 * np.pi * freq * t)
    pcm = (signal * 32767).astype(np.int16)

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm.tobytes())
    return buffer.getvalue()


def _write_wav(path: Path, freq: float) -> None:
    path.write_bytes(_make_wav_bytes(freq))


def _reset_api_state() -> None:
    run_id = uuid4().hex
    api_server._TRACK_DB_PATH = f"test_ingest_api_{run_id}.db"
    api_server._VECTOR_INDEX_PATH = f"test_ingest_api_{run_id}_index"
    api_server._vector_index = VectorIndex(dimension=EMBEDDING_SIZE)
    api_server._track_store = TrackStore(db_path=api_server._TRACK_DB_PATH)
    api_server._fingerprint_db.clear()
    api_server._job_queue = JobQueue()
    api_server._worker = BackgroundWorker(
        queue=api_server._job_queue,
        vector_index=api_server._vector_index,
        track_store=api_server._track_store,
        fingerprint_db=api_server._fingerprint_db,
        vector_index_path=api_server._VECTOR_INDEX_PATH,
    )
    api_server._ingestion_tracker = IngestionProgressTracker()


def test_scan_folder_and_enqueue_all_audio_files():
    root = Path(f"test_dataset_{uuid4().hex}")
    root.mkdir(parents=True, exist_ok=True)
    try:
        _write_wav(root / "a.wav", 440.0)
        _write_wav(root / "b.wav", 660.0)
        (root / "ignore.txt").write_text("x", encoding="utf-8")

        files = scan_folder(str(root))
        assert len(files) == 2

        queue = JobQueue()
        tracker = IngestionProgressTracker()
        queued, failed = enqueue_all_audio_files(
            path=str(root),
            queue=queue,
            embedding_path="vector_base",
            progress_tracker=tracker,
        )

        assert queued == 2
        assert failed == 0
        assert queue.size() == 2
        status = tracker.snapshot()
        assert status["total_files"] == 2
        assert status["queued_files"] == 2
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_ingest_endpoints_update_progress_status():
    _reset_api_state()

    root = Path(f"test_dataset_{uuid4().hex}")
    root.mkdir(parents=True, exist_ok=True)
    try:
        _write_wav(root / "a.wav", 440.0)
        _write_wav(root / "b.wav", 660.0)

        ingest_response = api_server.ingest_dataset(path=str(root))
        assert ingest_response["queued_files"] == 2
        assert ingest_response["status"]["total_files"] == 2

        status_before = api_server.ingest_status()
        assert status_before["processed_files"] == 0

        assert api_server._worker.process_one()
        assert api_server._worker.process_one()

        status_after = api_server.ingest_status()
        assert status_after["processed_files"] == 2
        assert status_after["queued_files"] == 0
    finally:
        shutil.rmtree(root, ignore_errors=True)
