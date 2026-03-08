"""Track embedding trainer from interaction streams."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np


class EmbeddingTrainer:
    def __init__(self, output_path: str = "models/track_embeddings.npy", dim: int = 32):
        self.output_path = output_path
        self.dim = int(max(4, dim))
        self.track_embeddings: Dict[str, np.ndarray] = {}

    def _rand_vec(self, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        vec = rng.normal(0, 1, size=self.dim).astype(np.float32)
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec /= norm
        return vec

    def train_track_embeddings(self, interactions: List[Dict[str, object]]) -> Dict[str, np.ndarray]:
        by_track: Dict[str, List[float]] = defaultdict(list)
        for item in interactions:
            tid = str(item.get("track_id") or "")
            if not tid:
                continue
            reward = float(item.get("reward", 1.0))
            by_track[tid].append(reward)

        embeddings: Dict[str, np.ndarray] = {}
        for tid, rewards in by_track.items():
            base = self._rand_vec(abs(hash(tid)) % (2**32))
            scale = float(np.mean(rewards)) if rewards else 1.0
            vec = base * max(0.1, min(2.0, scale))
            norm = float(np.linalg.norm(vec))
            if norm > 0:
                vec /= norm
            embeddings[tid] = vec.astype(np.float32)

        self.track_embeddings = embeddings
        return embeddings

    def update_embeddings_incrementally(self, interactions: List[Dict[str, object]]) -> Dict[str, np.ndarray]:
        if not self.track_embeddings:
            return self.train_track_embeddings(interactions)

        for item in interactions:
            tid = str(item.get("track_id") or "")
            if not tid:
                continue
            reward = float(item.get("reward", 1.0))
            if tid not in self.track_embeddings:
                self.track_embeddings[tid] = self._rand_vec(abs(hash(tid)) % (2**32))
            vec = self.track_embeddings[tid]
            vec = vec * (1.0 + 0.05 * reward)
            norm = float(np.linalg.norm(vec))
            if norm > 0:
                vec /= norm
            self.track_embeddings[tid] = vec.astype(np.float32)

        return self.track_embeddings

    def save_embeddings(self) -> str:
        Path(self.output_path).parent.mkdir(parents=True, exist_ok=True)
        if not self.track_embeddings:
            np.save(self.output_path, np.empty((0, self.dim), dtype=np.float32))
            return self.output_path

        track_ids = sorted(self.track_embeddings.keys())
        matrix = np.vstack([self.track_embeddings[tid] for tid in track_ids]).astype(np.float32)
        np.save(self.output_path, matrix)
        Path(self.output_path + ".ids").write_text("\n".join(track_ids), encoding="utf-8")
        return self.output_path
