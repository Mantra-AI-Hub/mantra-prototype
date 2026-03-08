import io
import shutil
import wave
from pathlib import Path
from uuid import uuid4

import numpy as np

import mantra.interfaces.api.api_server as api_server
from mantra.database.track_store import TrackStore
from mantra.ingestion.progress_tracker import IngestionProgressTracker
from mantra.pipeline.job_queue import JobQueue
from mantra.pipeline.worker import BackgroundWorker
from mantra.search_engine import clear_search_cache
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


def _reset_api_state() -> None:
    run_id = uuid4().hex
    api_server._TRACK_DB_PATH = f"test_batch_search_{run_id}.db"
    api_server._VECTOR_INDEX_PATH = f"test_batch_search_{run_id}_index"
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
    clear_search_cache()


def test_batch_search_returns_results_per_query():
    _reset_api_state()
    tmp_dir = Path(f"test_batch_search_{uuid4().hex}")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    try:
        api_server.index_audio(audio_bytes=_make_wav_bytes(440.0), filename="a.wav", track_id="a")
        api_server.index_audio(audio_bytes=_make_wav_bytes(660.0), filename="b.wav", track_id="b")
        assert api_server._worker.process_one()
        assert api_server._worker.process_one()

        q1 = tmp_dir / "q1.wav"
        q2 = tmp_dir / "q2.wav"
        q1.write_bytes(_make_wav_bytes(440.0))
        q2.write_bytes(_make_wav_bytes(660.0))

        payload = api_server.BatchSearchRequest(
            queries=[{"query_audio_path": str(q1)}, {"query_audio_path": str(q2)}],
            top_k=5,
        )
        response = api_server.search_batch(payload)

        assert len(response["results"]) == 2
        assert response["results"][0]["results"]
        assert response["results"][1]["results"]
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
