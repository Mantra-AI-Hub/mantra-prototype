"""Shazam-style audio fingerprint generation and matching."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

import numpy as np


AudioData = Tuple[np.ndarray, int]
Fingerprint = List[Tuple[str, int]]


def _frame_signal(samples: np.ndarray, frame_size: int = 2048, hop_size: int = 512) -> np.ndarray:
    if samples.size < frame_size:
        padded = np.zeros(frame_size, dtype=np.float32)
        padded[: samples.size] = samples
        return padded[None, :]

    frame_count = 1 + (samples.size - frame_size) // hop_size
    indices = np.arange(frame_size)[None, :] + hop_size * np.arange(frame_count)[:, None]
    return samples[indices]


def _magnitude_spectrogram(samples: np.ndarray, sample_rate: int, frame_size: int = 2048, hop_size: int = 512):
    frames = _frame_signal(samples, frame_size=frame_size, hop_size=hop_size)
    window = np.hanning(frame_size).astype(np.float32)
    stft = np.fft.rfft(frames * window[None, :], axis=1)
    magnitude = np.abs(stft).astype(np.float32)
    freqs = np.fft.rfftfreq(frame_size, d=1.0 / sample_rate).astype(np.float32)
    return magnitude, freqs


def _spectrogram_peaks(magnitude: np.ndarray, max_peaks_per_frame: int = 5) -> List[Tuple[int, int, float]]:
    if magnitude.size == 0:
        return []

    threshold = float(np.percentile(magnitude, 75))
    peaks: List[Tuple[int, int, float]] = []

    for t in range(magnitude.shape[0]):
        frame = magnitude[t]
        top = min(max_peaks_per_frame, frame.size)
        if top <= 0:
            continue
        candidates = np.argpartition(frame, -top)[-top:]
        for f in candidates:
            amp = float(frame[f])
            if amp >= threshold:
                peaks.append((t, int(f), amp))

    peaks.sort(key=lambda x: (x[0], -x[2]))
    return peaks


def generate_fingerprint(audio: AudioData) -> Fingerprint:
    """Generate landmark-hash fingerprint from audio samples."""
    if not isinstance(audio, tuple) or len(audio) != 2:
        raise ValueError("audio must be a tuple: (samples, sample_rate)")

    samples, sample_rate = audio
    samples = np.asarray(samples, dtype=np.float32).reshape(-1)
    sample_rate = int(sample_rate)

    magnitude, _ = _magnitude_spectrogram(samples, sample_rate)
    peaks = _spectrogram_peaks(magnitude)

    fanout = 5
    min_dt = 1
    max_dt = 40

    fingerprints: Fingerprint = []
    for i, (t1, f1, _) in enumerate(peaks):
        for j in range(1, fanout + 1):
            if i + j >= len(peaks):
                break
            t2, f2, _ = peaks[i + j]
            dt = t2 - t1
            if dt < min_dt:
                continue
            if dt > max_dt:
                break
            hash_key = f"{f1}:{f2}:{dt}"
            fingerprints.append((hash_key, int(t1)))

    return fingerprints


def _group_by_hash(fingerprint: Sequence[Tuple[str, int]]) -> Dict[str, List[int]]:
    grouped: Dict[str, List[int]] = defaultdict(list)
    for hash_key, time_bin in fingerprint:
        grouped[str(hash_key)].append(int(time_bin))
    return grouped


def match_fingerprint(
    query_fp: Sequence[Tuple[str, int]], database_fps: Dict[str, Sequence[Tuple[str, int]]]
) -> List[Tuple[str, float]]:
    """Match query fingerprint against database and return ranked similarity scores."""
    if not query_fp:
        return []

    query_map = _group_by_hash(query_fp)
    query_size = max(1, len(query_fp))
    ranked: List[Tuple[str, float]] = []

    for track_id, db_fp in database_fps.items():
        db_map = _group_by_hash(db_fp)
        offsets: Dict[int, int] = defaultdict(int)

        for hash_key, query_times in query_map.items():
            db_times = db_map.get(hash_key)
            if not db_times:
                continue
            for qt in query_times:
                for dt in db_times:
                    offsets[dt - qt] += 1

        best_alignment = max(offsets.values()) if offsets else 0
        if best_alignment <= 0:
            continue

        score = float(best_alignment) / float(query_size)
        ranked.append((str(track_id), score))

    ranked.sort(key=lambda item: item[1], reverse=True)
    return ranked
