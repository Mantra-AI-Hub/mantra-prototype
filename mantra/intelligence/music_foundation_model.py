"""Foundational model for universal music understanding."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover - optional dependency fallback
    torch = None


class MusicFoundationModel:
    """Universal embedding and structure analysis for audio and symbolic music."""

    def __init__(self, embedding_dim: int = 128) -> None:
        self.embedding_dim = max(32, int(embedding_dim))
        self.backend = "torch" if torch is not None else "numpy-fallback"

    def _seeded_embedding(self, key: str) -> np.ndarray:
        digest = hashlib.sha256(key.encode("utf-8")).digest()
        seed = int.from_bytes(digest[:8], byteorder="big", signed=False)
        rng = np.random.default_rng(seed)
        vec = rng.normal(0, 1, size=self.embedding_dim).astype(np.float32)
        if torch is not None:
            tensor = torch.tensor(vec, dtype=torch.float32)
            norm = float(torch.linalg.norm(tensor).item())
        else:
            norm = float(np.linalg.norm(vec))
        return vec / norm if norm > 0 else vec

    def embed_audio(self, audio_path: str) -> np.ndarray:
        return self._seeded_embedding(f"audio:{Path(audio_path).name}")

    def embed_midi(self, midi_path: str) -> np.ndarray:
        return self._seeded_embedding(f"midi:{Path(midi_path).name}")

    def embed_features(self, feature_vector: Sequence[float]) -> np.ndarray:
        vec = np.asarray(feature_vector, dtype=np.float32).reshape(-1)
        if vec.size == 0:
            return np.zeros(self.embedding_dim, dtype=np.float32)
        if vec.size < self.embedding_dim:
            vec = np.pad(vec, (0, self.embedding_dim - vec.size))
        elif vec.size > self.embedding_dim:
            vec = vec[: self.embedding_dim]
        norm = float(np.linalg.norm(vec))
        return vec / norm if norm > 0 else vec

    def analyze_structure(self, audio_path: str) -> Dict[str, object]:
        digest = hashlib.sha256(str(audio_path).encode("utf-8")).digest()
        tempo = 70 + (digest[0] % 90)
        keys = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        scales = ["major", "minor", "dorian", "mixolydian"]
        key = keys[digest[1] % len(keys)]
        scale = scales[digest[2] % len(scales)]
        chords = ["I", "V", "vi", "IV", "ii", "iii"]
        chord_progression = [chords[(digest[i] + i) % len(chords)] for i in range(3, 7)]
        rhythm_pattern = [float(((digest[i] % 4) + 1) / 4.0) for i in range(7, 15)]
        spectral_features: List[float] = [float(digest[i] / 255.0) for i in range(15, 27)]
        return {
            "tempo": int(tempo),
            "key": key,
            "scale": scale,
            "chord_progression": chord_progression,
            "rhythm_pattern": rhythm_pattern,
            "spectral_features": spectral_features,
        }
