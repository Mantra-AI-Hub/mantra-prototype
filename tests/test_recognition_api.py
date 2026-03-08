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
    api_server._TRACK_DB_PATH = f"test_recognize_{run_id}.db"
    api_server._VECTOR_INDEX_PATH = f"test_recognize_{run_id}_index"
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


def test_recognition_api_and_stream_recognition():
    _reset_api_state()
    root = Path(f"test_recognize_{uuid4().hex}")
    root.mkdir(parents=True, exist_ok=True)

    try:
        base = _make_wav_bytes(440.0)
        api_server.index_audio(audio_bytes=base, filename="song.wav", track_id="song_1")
        assert api_server._worker.process_one()

        qpath = root / "query.wav"
        qpath.write_bytes(_make_wav_bytes(440.0))

        response = api_server.recognize(api_server.RecognizeRequest(audio_path=str(qpath)))
        assert response["track_id"] == "song_1"
        assert response["score"] > 0

        # stream path
        samples = np.frombuffer(_make_wav_bytes(440.0)[44:], dtype=np.int16).astype(np.float32) / 32768.0
        stream_resp = api_server.recognize_stream(
            api_server.StreamRecognizeRequest(samples=samples[:4096].tolist(), sample_rate=22050)
        )
        assert "track_id" in stream_resp
    finally:
        shutil.rmtree(root, ignore_errors=True)
