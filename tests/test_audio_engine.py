import wave
from pathlib import Path

import numpy as np
import pytest

from mantra.audio_engine.audio_loader import load_audio
from mantra.audio_engine.feature_extractor import (
    PITCH_CONTOUR_LENGTH,
    extract_chroma,
    extract_pitch_contour,
    extract_spectral_features,
    extract_tempo,
)


def _write_test_wav(path: Path, sample_rate: int = 22050, duration_sec: float = 0.5, frequency: float = 440.0) -> None:
    t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), endpoint=False, dtype=np.float32)
    signal = 0.5 * np.sin(2 * np.pi * frequency * t)
    pcm = (signal * 32767).astype(np.int16)

    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.tobytes())


def test_load_audio_wav_success():
    wav_path = Path("test_audio_tone.wav")
    _write_test_wav(wav_path)
    try:
        samples, sample_rate = load_audio(str(wav_path))
        assert isinstance(samples, np.ndarray)
        assert samples.ndim == 1
        assert samples.size > 0
        assert sample_rate == 22050
    finally:
        wav_path.unlink(missing_ok=True)


def test_load_audio_rejects_non_wav():
    midi_like_path = Path("test_audio_invalid.mid")
    midi_like_path.write_bytes(b"MThd")
    try:
        with pytest.raises(ValueError):
            load_audio(str(midi_like_path))
    finally:
        midi_like_path.unlink(missing_ok=True)


def test_extractors_return_expected_shapes():
    wav_path = Path("test_audio_tone.wav")
    _write_test_wav(wav_path)
    try:
        audio = load_audio(str(wav_path))
        chroma = extract_chroma(audio)
        tempo = extract_tempo(audio)
        contour = extract_pitch_contour(audio)
        spectral = extract_spectral_features(audio)

        assert chroma.shape == (12,)
        assert np.isfinite(chroma).all()

        assert isinstance(tempo, float)
        assert np.isfinite(tempo)

        assert contour.shape == (PITCH_CONTOUR_LENGTH,)
        assert np.isfinite(contour).all()

        assert len(spectral) == 10
        assert all(np.isfinite(v) for v in spectral.values())
    finally:
        wav_path.unlink(missing_ok=True)
