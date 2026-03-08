"""Feature extraction utilities for raw audio."""

from typing import Dict, Tuple

import numpy as np


AudioData = Tuple[np.ndarray, int]
PITCH_CONTOUR_LENGTH = 64


def _unpack_audio(audio: AudioData) -> Tuple[np.ndarray, int]:
    if not isinstance(audio, tuple) or len(audio) != 2:
        raise ValueError("audio must be a tuple: (samples, sample_rate)")
    samples, sample_rate = audio
    return np.asarray(samples, dtype=np.float32), int(sample_rate)


def _safe_mean(values: np.ndarray) -> float:
    return float(np.mean(values)) if values.size else 0.0


def _safe_std(values: np.ndarray) -> float:
    return float(np.std(values)) if values.size else 0.0


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
    spectrum = np.fft.rfft(frames * window[None, :], axis=1)
    magnitude = np.abs(spectrum).astype(np.float32)
    freqs = np.fft.rfftfreq(frame_size, d=1.0 / sample_rate).astype(np.float32)
    return magnitude, freqs


def extract_chroma(audio: AudioData) -> np.ndarray:
    """Extract a 12-bin chroma summary vector from FFT magnitudes."""
    samples, sample_rate = _unpack_audio(audio)
    magnitude, freqs = _magnitude_spectrogram(samples, sample_rate)
    energy = magnitude.sum(axis=0)

    chroma = np.zeros(12, dtype=np.float32)
    valid = freqs > 0
    valid_freqs = freqs[valid]
    valid_energy = energy[valid]
    if valid_freqs.size == 0:
        return chroma

    midi = 69.0 + 12.0 * np.log2(valid_freqs / 440.0)
    pitch_classes = np.mod(np.round(midi).astype(int), 12)
    for pitch_class, value in zip(pitch_classes, valid_energy):
        chroma[pitch_class] += float(value)

    total = float(chroma.sum())
    if total > 0:
        chroma /= total
    return chroma


def extract_tempo(audio: AudioData) -> float:
    """Estimate tempo using RMS-onset autocorrelation in BPM."""
    samples, sample_rate = _unpack_audio(audio)
    frames = _frame_signal(samples, frame_size=1024, hop_size=512)
    rms = np.sqrt(np.mean(frames * frames, axis=1)).astype(np.float32)
    onset = np.maximum(0.0, np.diff(rms, prepend=rms[:1]))

    if onset.size < 4 or float(onset.max()) == 0.0:
        return 0.0

    onset = onset - np.mean(onset)
    autocorr = np.correlate(onset, onset, mode="full")[onset.size - 1 :]
    min_bpm, max_bpm = 40.0, 220.0
    min_lag = int(max(1, np.floor((60.0 * sample_rate) / (max_bpm * 512))))
    max_lag = int(max(min_lag + 1, np.ceil((60.0 * sample_rate) / (min_bpm * 512))))
    max_lag = min(max_lag, autocorr.size - 1)

    if max_lag <= min_lag:
        return 0.0

    candidate = autocorr[min_lag : max_lag + 1]
    lag = int(np.argmax(candidate)) + min_lag
    if lag <= 0:
        return 0.0
    return float((60.0 * sample_rate) / (lag * 512))


def extract_pitch_contour(audio: AudioData) -> np.ndarray:
    """Extract a fixed-length normalized dominant-frequency contour."""
    samples, sample_rate = _unpack_audio(audio)
    magnitude, freqs = _magnitude_spectrogram(samples, sample_rate, frame_size=1024, hop_size=256)
    if magnitude.size == 0:
        return np.zeros(PITCH_CONTOUR_LENGTH, dtype=np.float32)

    dominant_bins = np.argmax(magnitude, axis=1)
    dominant_freqs = freqs[dominant_bins].astype(np.float32)
    target_x = np.linspace(0.0, 1.0, num=PITCH_CONTOUR_LENGTH, dtype=np.float32)
    source_x = np.linspace(0.0, 1.0, num=dominant_freqs.size, dtype=np.float32)
    contour = np.interp(target_x, source_x, dominant_freqs).astype(np.float32)

    max_value = float(np.max(np.abs(contour)))
    if max_value > 0:
        contour /= max_value
    return contour


def extract_spectral_features(audio: AudioData) -> Dict[str, float]:
    """Extract compact spectral statistics for embedding construction."""
    samples, sample_rate = _unpack_audio(audio)
    magnitude, freqs = _magnitude_spectrogram(samples, sample_rate)
    energy = np.maximum(magnitude.sum(axis=1), 1e-8)

    centroid = (magnitude * freqs[None, :]).sum(axis=1) / energy
    spread = np.sqrt(((freqs[None, :] - centroid[:, None]) ** 2 * magnitude).sum(axis=1) / energy)

    cumulative = np.cumsum(magnitude, axis=1)
    rolloff_threshold = 0.85 * cumulative[:, -1:]
    rolloff_bins = np.argmax(cumulative >= rolloff_threshold, axis=1)
    rolloff = freqs[rolloff_bins]

    signs = np.sign(samples)
    zcr = np.abs(np.diff(signs, prepend=signs[:1]))
    zcr_rate = np.mean(zcr > 0) if zcr.size else 0.0

    rms = np.sqrt(np.mean(_frame_signal(samples, frame_size=1024, hop_size=512) ** 2, axis=1))

    return {
        "centroid_mean": _safe_mean(centroid),
        "centroid_std": _safe_std(centroid),
        "bandwidth_mean": _safe_mean(spread),
        "bandwidth_std": _safe_std(spread),
        "rolloff_mean": _safe_mean(rolloff),
        "rolloff_std": _safe_std(rolloff),
        "zcr_mean": float(zcr_rate),
        "zcr_std": 0.0,
        "rms_mean": _safe_mean(rms),
        "rms_std": _safe_std(rms),
    }
