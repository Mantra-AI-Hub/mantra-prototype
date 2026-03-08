import io
import wave
from pathlib import Path
from uuid import uuid4

import numpy as np

from mantra.database.track_store import TrackStore
from mantra.pipeline.job_queue import JobQueue
from mantra.pipeline.tasks import embedding_task, fingerprint_task
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


def _decode_wav_bytes(raw_bytes: bytes):
    with wave.open(io.BytesIO(raw_bytes), "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_rate = wav_file.getframerate()
        sample_width = wav_file.getsampwidth()
        frames = wav_file.readframes(wav_file.getnframes())

    if sample_width == 2:
        dtype = np.int16
        offset = 0.0
        scale = 32768.0
    else:
        raise ValueError("unexpected sample width")

    audio = np.frombuffer(frames, dtype=dtype).astype(np.float32)
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    audio = (audio - offset) / scale
    return audio.astype(np.float32), int(sample_rate)


def test_job_queue_fifo_behavior():
    queue = JobQueue()

    a = queue.enqueue("task_a", {"x": 1})
    b = queue.enqueue("task_b", {"x": 2})

    first = queue.dequeue()
    second = queue.dequeue()
    third = queue.dequeue()

    assert first["job_id"] == a["job_id"]
    assert second["job_id"] == b["job_id"]
    assert third is None


def test_tasks_build_embedding_and_fingerprint():
    audio = _decode_wav_bytes(_make_wav_bytes(440.0))

    embedding = embedding_task(audio)
    fingerprint = fingerprint_task(audio)

    assert embedding.shape == (EMBEDDING_SIZE,)
    assert np.isfinite(embedding).all()
    assert fingerprint


def test_worker_processes_index_audio_job():
    run_id = uuid4().hex
    db_path = f"test_pipeline_{run_id}.db"
    vector_path = f"test_pipeline_{run_id}_index"

    queue = JobQueue()
    store = TrackStore(db_path=db_path)
    vector_index = VectorIndex(dimension=EMBEDDING_SIZE)
    fingerprint_db = {}

    worker = BackgroundWorker(
        queue=queue,
        vector_index=vector_index,
        track_store=store,
        fingerprint_db=fingerprint_db,
        vector_index_path=vector_path,
    )

    queue.enqueue(
        "index_audio",
        {
            "track_id": "track_pipeline",
            "filename": "pipeline.wav",
            "embedding_path": vector_path,
            "audio_bytes": _make_wav_bytes(440.0),
        },
    )

    assert worker.process_one()

    record = store.get_track("track_pipeline")
    assert record is not None
    assert record["fingerprint_hash_count"] > 0

    search = vector_index.search(np.ones(EMBEDDING_SIZE, dtype=np.float32), top_k=1)
    assert search
    assert search[0][0] == "track_pipeline"

    for suffix in (".faiss", ".meta.json", ".npz"):
        Path(vector_path + suffix).unlink(missing_ok=True)
