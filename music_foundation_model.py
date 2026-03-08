"""Unified multimodal music foundation model surface with lightweight fallback."""

from __future__ import annotations

import hashlib
from typing import Dict, Iterable, List

import numpy as np


class MusicFoundationModel:
    def __init__(self, dim: int = 192):
        self.dim = max(32, int(dim))
        self.is_trained = False
        self.backend = "transformer-fallback"

    def _hash_vec(self, key: str) -> np.ndarray:
        digest = hashlib.sha256(key.encode("utf-8")).digest()
        seed = int.from_bytes(digest[:8], byteorder="big", signed=False)
        rng = np.random.default_rng(seed)
        vec = rng.normal(0, 1, size=self.dim).astype(np.float32)
        norm = float(np.linalg.norm(vec))
        return vec / norm if norm > 0 else vec

    def encode_audio(self, audio_ref: str) -> np.ndarray:
        return self._hash_vec(f"audio:{audio_ref}")

    def encode_lyrics(self, lyrics: str) -> np.ndarray:
        return self._hash_vec(f"lyrics:{lyrics or ''}")

    def encode_metadata(self, metadata: Dict[str, object] | None) -> np.ndarray:
        return self._hash_vec(f"meta:{sorted((metadata or {}).items())}")

    def encode_artist_graph(self, artist_id: str, neighbors: Iterable[str] | None = None) -> np.ndarray:
        serialized = ",".join(sorted(str(x) for x in (neighbors or [])))
        return self._hash_vec(f"artist:{artist_id}:{serialized}")

    def build_multimodal_embedding(
        self,
        audio: str,
        lyrics: str = "",
        metadata: Dict[str, object] | None = None,
        artist_id: str | None = None,
        artist_neighbors: Iterable[str] | None = None,
    ) -> np.ndarray:
        audio_vec = self.encode_audio(audio)
        lyric_vec = self.encode_lyrics(lyrics)
        meta_vec = self.encode_metadata(metadata)
        artist_vec = self.encode_artist_graph(artist_id or "", artist_neighbors)
        mixed = (0.45 * audio_vec + 0.2 * lyric_vec + 0.2 * meta_vec + 0.15 * artist_vec).astype(np.float32)
        norm = float(np.linalg.norm(mixed))
        return mixed / norm if norm > 0 else mixed

    def train_or_finetune(self, dataset: List[Dict[str, object]]) -> Dict[str, object]:
        self.is_trained = True
        return {"backend": self.backend, "samples": len(dataset), "trained": True}

