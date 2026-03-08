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
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.tobytes())
    return buf.getvalue()


def _reset_api_state() -> None:
    run_id = uuid4().hex
    api_server._TRACK_DB_PATH = f"test_cross_modal_{run_id}.db"
    api_server._VECTOR_INDEX_PATH = f"test_cross_modal_{run_id}_index"
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


def test_cross_modal_text_and_lyrics_endpoints():
    _reset_api_state()
    api_server.index_audio(
        audio_bytes=_make_wav_bytes(440.0),
        filename="ambient.wav",
        track_id="ambient_1",
        genre="ambient",
        tags="calm,space",
    )
    assert api_server._worker.process_one()

    text_resp = api_server.search_text(api_server.TextSearchRequest(query="ambient calm music", top_k=5))
    lyr_resp = api_server.search_lyrics(api_server.TextSearchRequest(query="calm space", top_k=5))

    assert text_resp["results"]
    assert lyr_resp["results"]
    assert text_resp["results"][0]["track_id"] == "ambient_1"
