"""Shazam-style fingerprint generation utilities."""

from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Tuple
import wave

import numpy as np


Fingerprint = Tuple[str, int]


def _load_wav(audio_path: str) -> Tuple[np.ndarray, int]:
    path = Path(audio_path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    if path.suffix.lower() != ".wav":
        raise ValueError("Only WAV files are supported")

    with wave.open(str(path), "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_rate = wav_file.getframerate()
        sample_width = wav_file.getsampwidth()
        frames = wav_file.readframes(wav_file.getnframes())

    if sample_width == 1:
        dtype, offset, scale = np.uint8, 128.0, 128.0
    elif sample_width == 2:
        dtype, offset, scale = np.int16, 0.0, 32768.0
    elif sample_width == 4:
        dtype, offset, scale = np.int32, 0.0, 2147483648.0
    else:
        raise ValueError(f"Unsupported WAV sample width: {sample_width}")

    audio = np.frombuffer(frames, dtype=dtype).astype(np.float32)
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    audio = (audio - offset) / scale
    return audio.astype(np.float32), int(sample_rate)


def _spectrogram(audio: np.ndarray, frame_size: int = 2048, hop_size: int = 512) -> np.ndarray:
    if audio.size < frame_size:
        padded = np.zeros(frame_size, dtype=np.float32)
        padded[: audio.size] = audio
        frames = padded[None, :]
    else:
        frame_count = 1 + (audio.size - frame_size) // hop_size
        idx = np.arange(frame_size)[None, :] + hop_size * np.arange(frame_count)[:, None]
        frames = audio[idx]
    window = np.hanning(frame_size).astype(np.float32)
    spectrum = np.fft.rfft(frames * window[None, :], axis=1)
    return np.abs(spectrum).astype(np.float32)


def _extract_peaks(spec: np.ndarray, max_peaks_per_frame: int = 5) -> List[Tuple[int, int]]:
    if spec.size == 0:
        return []
    threshold = float(np.percentile(spec, 80))
    peaks: List[Tuple[int, int]] = []
    for t in range(spec.shape[0]):
        row = spec[t]
        top = min(max_peaks_per_frame, row.size)
        bins = np.argpartition(row, -top)[-top:]
        for f in bins:
            if float(row[f]) >= threshold:
                peaks.append((t, int(f)))
    return peaks


def _pair_peaks(peaks: Sequence[Tuple[int, int]], fanout: int = 5, max_dt: int = 40) -> List[Fingerprint]:
    fingerprints: List[Fingerprint] = []
    for i, (t1, f1) in enumerate(peaks):
        for j in range(1, fanout + 1):
            if i + j >= len(peaks):
                break
            t2, f2 = peaks[i + j]
            dt = t2 - t1
            if dt <= 0:
                continue
            if dt > max_dt:
                break
            fingerprints.append((f"{f1}:{f2}:{dt}", int(t1)))
    return fingerprints


def generate_fingerprints_from_audio(audio: Tuple[np.ndarray, int]) -> List[Fingerprint]:
    samples, _sample_rate = audio
    samples = np.asarray(samples, dtype=np.float32).reshape(-1)
    spec = _spectrogram(samples)
    peaks = _extract_peaks(spec)
    return _pair_peaks(peaks)


def generate_fingerprints(audio_path: str) -> List[Fingerprint]:
    audio = _load_wav(audio_path)
    return generate_fingerprints_from_audio(audio)
