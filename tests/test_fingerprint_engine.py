import io
import shutil
import wave
from pathlib import Path
from uuid import uuid4

import numpy as np

from mantra.fingerprint_engine import generate_fingerprints, generate_fingerprints_from_audio


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


def test_generate_fingerprints_pipeline():
    root = Path(f"test_fp_engine_{uuid4().hex}")
    root.mkdir(parents=True, exist_ok=True)
    try:
        wav_path = root / "q.wav"
        _make_wav(wav_path, 440.0)

        fp1 = generate_fingerprints(str(wav_path))
        assert fp1

        samples = np.frombuffer(wav_path.read_bytes()[-44100:], dtype=np.int16).astype(np.float32) / 32768.0
        fp2 = generate_fingerprints_from_audio((samples, 22050))
        assert isinstance(fp2, list)
    finally:
        shutil.rmtree(root, ignore_errors=True)
