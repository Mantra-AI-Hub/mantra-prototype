"""Storage and similarity retrieval for music genomes."""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from mantra.intelligence.music_genome_engine import MusicGenomeEngine
from mantra.vector_index import VectorIndex


class MusicGenomeStore:
    def __init__(self, vector_index: VectorIndex | None = None, genome_engine: MusicGenomeEngine | None = None) -> None:
        self.genome_engine = genome_engine or MusicGenomeEngine()
        self.dim = 10
        self.vector_index = vector_index or VectorIndex(dimension=self.dim)
        self.genomes: Dict[str, Dict[str, object]] = {}

    def _to_vector(self, genome: Dict[str, object]) -> np.ndarray:
        vec = np.array(
            [
                float(genome.get("energy", 0.0)),
                float(genome.get("danceability", 0.0)),
                float(genome.get("tempo", 0.0)) / 200.0,
                float(genome.get("rhythm_complexity", 0.0)),
                float(genome.get("harmonic_complexity", 0.0)),
                float(genome.get("melodic_density", 0.0)),
                float(genome.get("vocal_presence", 0.0)),
                float(hash(str(genome.get("genre", ""))) % 100) / 100.0,
                float(hash(str(genome.get("mood", ""))) % 100) / 100.0,
                float(hash(str(genome.get("production_style", ""))) % 100) / 100.0,
            ],
            dtype=np.float32,
        )
        norm = float(np.linalg.norm(vec))
        return vec / norm if norm > 0 else vec

    def store_genome(self, track_id: str, genome: Dict[str, object]) -> None:
        track_key = str(track_id)
        self.genomes[track_key] = dict(genome)
        vec = self._to_vector(genome)
        try:
            self.vector_index.add(vec, track_key)
        except Exception:
            # Rebuild fallback when index does not support updates for duplicate IDs.
            self.vector_index = VectorIndex(dimension=self.dim)
            for tid, g in self.genomes.items():
                self.vector_index.add(self._to_vector(g), tid)

    def get_genome(self, track_id: str) -> Optional[Dict[str, object]]:
        value = self.genomes.get(str(track_id))
        return dict(value) if value else None

    def search_similar(self, genome: Dict[str, object], top_k: int = 5) -> List[Dict[str, object]]:
        query = self._to_vector(genome)
        candidates = self.vector_index.search(query, int(top_k))
        results: List[Dict[str, object]] = []
        for track_id, score in candidates:
            stored = self.genomes.get(str(track_id))
            if not stored:
                continue
            genome_score = self.genome_engine.similarity_score(genome, stored)
            results.append(
                {
                    "track_id": str(track_id),
                    "score": float(score),
                    "genome_similarity": float(genome_score),
                }
            )
        results.sort(key=lambda item: (item["genome_similarity"] + item["score"]) / 2.0, reverse=True)
        return results[: max(0, int(top_k))]
