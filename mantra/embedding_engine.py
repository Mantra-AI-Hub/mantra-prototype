"""Optional GPU-accelerated embedding computation."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np

from mantra.audio_engine.audio_loader import load_audio
from mantra.audio_engine.feature_extractor import (
    extract_chroma,
    extract_pitch_contour,
    extract_spectral_features,
    extract_tempo,
)
from mantra.vector_engine.embedding_builder import build_music_embedding

try:
    import torch  # type: ignore
except Exception:  # pragma: no cover
    torch = None


class EmbeddingEngine:
    """Embedding engine with optional torch.cuda acceleration."""

    def __init__(self):
        self.model = None
        self.device = "cpu"
        self.gpu_enabled = False

    def load_model(self) -> Dict[str, object]:
        self.model = "mantra-embedding-v1"
        self.gpu_enabled = bool(torch is not None and torch.cuda.is_available())
        self.device = "cuda" if self.gpu_enabled else "cpu"
        return {"model": self.model, "device": self.device, "gpu_enabled": self.gpu_enabled}

    @staticmethod
    def _compute_features(audio_path: str) -> Dict[str, object]:
        audio = load_audio(audio_path)
        return {
            "chroma": extract_chroma(audio),
            "tempo": extract_tempo(audio),
            "pitch_contour": extract_pitch_contour(audio),
            "spectral": extract_spectral_features(audio),
        }

    def compute_embedding(self, audio_path: str) -> np.ndarray:
        if not Path(audio_path).exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        features = self._compute_features(audio_path)
        embedding = build_music_embedding(features)
        if self.gpu_enabled and torch is not None:
            _ = torch.from_numpy(embedding).to("cuda")
        return embedding

    def compute_embeddings_batch(self, audio_paths: List[str]) -> np.ndarray:
        embeddings = [self.compute_embedding(path) for path in audio_paths]
        if not embeddings:
            return np.empty((0, 0), dtype=np.float32)
        batch = np.vstack(embeddings).astype(np.float32)
        if self.gpu_enabled and torch is not None:
            _ = torch.from_numpy(batch).to("cuda")
        return batch
