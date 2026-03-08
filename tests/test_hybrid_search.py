import io
import shutil
import wave
from pathlib import Path
from uuid import uuid4

import numpy as np

from mantra.fingerprint_engine import generate_fingerprints
from mantra.fingerprint_index import FingerprintIndex
from mantra.hybrid_search import hybrid_search
from mantra.database.track_store import TrackStore
from mantra.vector_index import VectorIndex


def _make_wav(path: Path, freq: float, duration: float = 1.0, sample_rate: int = 22050):
    t = np.linspace(0.0, duration, int(sample_rate * duration), endpoint=False, dtype=np.float32)
    signal = 0.5 * np.sin(2.0 * np.pi * freq * t)
    pcm = (signal * 32767).astype(np.int16)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm.tobytes())
    path.write_bytes(buffer.getvalue())


def test_hybrid_search_fuses_fingerprint_and_embedding():
    root = Path(f"test_hybrid_{uuid4().hex}")
    root.mkdir(parents=True, exist_ok=True)

    try:
        a = root / "a.wav"
        q = root / "q.wav"
        _make_wav(a, 440.0)
        _make_wav(q, 440.0)

        db_path = str(root / "tracks.db")
        store = TrackStore(db_path=db_path)
        store.add_track(
            {
                "track_id": "track_a",
                "filename": "a.wav",
                "duration": 1.0,
                "embedding_path": str(root / "vec"),
                "fingerprint_hash_count": 10,
                "created_at": "2026-01-01T00:00:00+00:00",
                "artist": "unknown",
                "genre": "ambient",
                "album": "x",
                "tags": ["calm"],
                "year": 2020,
            }
        )

        index = VectorIndex(dimension=87)
        index.add(np.ones(87, dtype=np.float32), "track_a")
        vec_base = str(root / "vec")
        index.save(vec_base)

        fp_index = FingerprintIndex()
        fp_index.add("track_a", generate_fingerprints(str(a)))

        results = hybrid_search(
            query_audio_path=str(q),
            top_k=5,
            fingerprint_index=fp_index,
            vector_index_path=vec_base,
            track_store=store,
        )
        assert results
        assert results[0]["track_id"] == "track_a"
    finally:
        shutil.rmtree(root, ignore_errors=True)
