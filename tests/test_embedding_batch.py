import io
import shutil
import wave
from pathlib import Path
from uuid import uuid4

import numpy as np

from mantra.embedding_engine import EmbeddingEngine


def _make_wav(path: Path, freq: float, duration: float = 1.0, sample_rate: int = 22050) -> None:
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


def test_embedding_engine_batch_pipeline():
    root = Path(f"test_embedding_batch_{uuid4().hex}")
    root.mkdir(parents=True, exist_ok=True)
    try:
        p1 = root / "a.wav"
        p2 = root / "b.wav"
        _make_wav(p1, 440.0)
        _make_wav(p2, 660.0)

        engine = EmbeddingEngine()
        model_info = engine.load_model()
        assert "gpu_enabled" in model_info

        single = engine.compute_embedding(str(p1))
        batch = engine.compute_embeddings_batch([str(p1), str(p2)])

        assert single.ndim == 1
        assert batch.shape[0] == 2
        assert np.isfinite(batch).all()
    finally:
        shutil.rmtree(root, ignore_errors=True)
