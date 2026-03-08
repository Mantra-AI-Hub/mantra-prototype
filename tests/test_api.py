import io
import wave
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
    api_server._TRACK_DB_PATH = f"test_api_{run_id}.db"
    api_server._VECTOR_INDEX_PATH = f"test_api_{run_id}_index"
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


def test_health_endpoint():
    _reset_api_state()
    assert api_server.health() == {"status": "ok"}


def test_index_and_search_audio_flow():
    _reset_api_state()
    audio_bytes = _make_wav_bytes(440.0)
    index_payload = api_server.index_audio(
        audio_bytes=audio_bytes,
        filename="track_a.wav",
        track_id="track_a",
    )
    assert index_payload["track_id"] == "track_a"
    assert index_payload["metadata"]["filename"] == "track_a.wav"
    assert api_server._worker.process_one()

    search_payload = api_server.search_audio(
        audio_bytes=audio_bytes,
        filename="query.wav",
        top_k=3,
    )
    assert search_payload["results"]
    assert search_payload["results"][0]["track_id"] == "track_a"
    assert search_payload["results"][0]["metadata"]["track_id"] == "track_a"


def test_fingerprint_match_endpoint():
    _reset_api_state()

    base = _make_wav_bytes(440.0, duration=2.0)
    other = _make_wav_bytes(660.0, duration=2.0)

    api_server.index_audio(
        audio_bytes=base,
        filename="track_a.wav",
        track_id="track_a",
    )
    assert api_server._worker.process_one()
    api_server.index_audio(
        audio_bytes=other,
        filename="track_b.wav",
        track_id="track_b",
    )
    assert api_server._worker.process_one()

    payload = api_server.fingerprint_match(
        audio_bytes=base,
        filename="snippet.wav",
        top_k=2,
    )
    assert payload["results"]
    assert payload["results"][0]["track_id"] == "track_a"
    assert payload["results"][0]["metadata"]["filename"] == "track_a.wav"


def test_index_audio_uses_filename_stem_for_default_track_id():
    _reset_api_state()
    audio_bytes = _make_wav_bytes(440.0)
    index_payload = api_server.index_audio(
        audio_bytes=audio_bytes,
        filename="folder/subfolder/song_name.wav",
        track_id=None,
    )
    assert index_payload["track_id"] == "song_name"
