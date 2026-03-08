"""Audio loading utilities for waveform analysis."""

import wave
from pathlib import Path
from typing import Tuple

import numpy as np


AudioData = Tuple[np.ndarray, int]


def load_audio(file_path: str) -> AudioData:
    """Load a WAV file and return mono samples and sample rate."""
    path = Path(file_path)

    if path.suffix.lower() != ".wav":
        raise ValueError("Only .wav files are supported.")

    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    with wave.open(str(path), "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_rate = wav_file.getframerate()
        sample_width = wav_file.getsampwidth()
        frames = wav_file.readframes(wav_file.getnframes())

    if sample_width == 1:
        dtype = np.uint8
        offset = 128.0
        scale = 128.0
    elif sample_width == 2:
        dtype = np.int16
        offset = 0.0
        scale = 32768.0
    elif sample_width == 4:
        dtype = np.int32
        offset = 0.0
        scale = 2147483648.0
    else:
        raise ValueError(f"Unsupported WAV sample width: {sample_width} bytes")

    audio = np.frombuffer(frames, dtype=dtype).astype(np.float32)
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)

    audio = (audio - offset) / scale
    return audio.astype(np.float32, copy=False), int(sample_rate)
