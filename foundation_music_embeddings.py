"""Foundation-style audio/multimodal embeddings with deterministic fallbacks."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, Iterable, Sequence

import numpy as np


class FoundationMusicEmbeddings:
    def __init__(self, dim: int = 128, feature_store=None):
        self.dim = max(16, int(dim))
        self.model_name = "fallback"
        self.feature_store = feature_store

    def load_pretrained_audio_model(self, preferred: str | None = None) -> Dict[str, str]:
        # Placeholder for CLAP/AudioCLIP/MusicLM adapters.
        self.model_name = preferred or "fallback"
        return {"model": self.model_name, "status": "loaded"}

    def _hash_to_vec(self, key: str) -> np.ndarray:
        digest = hashlib.sha256(key.encode("utf-8")).digest()
        seed = int.from_bytes(digest[:8], byteorder="big", signed=False)
        rng = np.random.default_rng(seed)
        vec = rng.normal(0, 1, size=self.dim).astype(np.float32)
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec /= norm
        return vec

    def compute_audio_embedding(self, audio_path: str) -> np.ndarray:
        path = Path(audio_path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        vec = self._hash_to_vec(str(path.resolve()))
        return vec

    def compute_multimodal_embedding(
        self,
        audio: str | Sequence[float],
        lyrics: str = "",
        metadata: Dict[str, object] | None = None,
    ) -> np.ndarray:
        if isinstance(audio, str):
            audio_vec = self._hash_to_vec(audio)
        else:
            arr = np.asarray(list(audio), dtype=np.float32).reshape(-1)
            if arr.size == 0:
                audio_vec = np.zeros(self.dim, dtype=np.float32)
            else:
                padded = np.zeros(self.dim, dtype=np.float32)
                take = min(self.dim, arr.size)
                padded[:take] = arr[:take]
                norm = float(np.linalg.norm(padded))
                audio_vec = padded / norm if norm > 0 else padded

        lyric_vec = self._hash_to_vec(str(lyrics or ""))
        meta_key = str(sorted((metadata or {}).items()))
        meta_vec = self._hash_to_vec(meta_key)
        fused = (0.6 * audio_vec + 0.25 * lyric_vec + 0.15 * meta_vec).astype(np.float32)
        norm = float(np.linalg.norm(fused))
        if norm > 0:
            fused /= norm
        return fused

    def cache_embedding(self, entity_id: str, embedding: Iterable[float]) -> None:
        if self.feature_store is None:
            return
        current = self.feature_store.get_track_features(entity_id) or {}
        current["foundation_embedding"] = [float(v) for v in embedding]
        self.feature_store.store_track_features(entity_id, current)


