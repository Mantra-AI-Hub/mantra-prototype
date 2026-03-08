"""Build fixed-length vectors from extracted music features."""

from typing import Dict

import numpy as np


EMBEDDING_SIZE = 87
PITCH_CONTOUR_LENGTH = 64
SPECTRAL_KEYS = [
    "centroid_mean",
    "centroid_std",
    "bandwidth_mean",
    "bandwidth_std",
    "rolloff_mean",
    "rolloff_std",
    "zcr_mean",
    "zcr_std",
    "rms_mean",
    "rms_std",
]


def _as_float_vector(values: object, expected_size: int) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32).flatten()
    if array.size >= expected_size:
        return array[:expected_size]
    padded = np.zeros(expected_size, dtype=np.float32)
    padded[: array.size] = array
    return padded


def build_music_embedding(features: Dict[str, object]) -> np.ndarray:
    """Build a fixed-length numeric embedding for similarity search."""
    chroma = _as_float_vector(features.get("chroma", np.zeros(12, dtype=np.float32)), 12)

    tempo = features.get("tempo", 0.0)
    tempo_value = float(np.asarray(tempo).squeeze()) if np.size(tempo) else 0.0
    tempo_vector = np.array([tempo_value], dtype=np.float32)

    pitch_contour = _as_float_vector(
        features.get("pitch_contour", np.zeros(PITCH_CONTOUR_LENGTH, dtype=np.float32)),
        PITCH_CONTOUR_LENGTH,
    )

    spectral = features.get("spectral", {})
    if not isinstance(spectral, dict):
        spectral = {}
    spectral_vector = np.array([float(spectral.get(key, 0.0)) for key in SPECTRAL_KEYS], dtype=np.float32)

    embedding = np.concatenate([chroma, tempo_vector, pitch_contour, spectral_vector]).astype(np.float32)

    if embedding.size != EMBEDDING_SIZE:
        raise RuntimeError(f"Unexpected embedding length: {embedding.size}")

    return embedding
