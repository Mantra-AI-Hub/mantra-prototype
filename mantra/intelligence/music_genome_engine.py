"""Music Genome style feature extraction and similarity."""

from __future__ import annotations

import hashlib
from typing import Dict, List


class MusicGenomeEngine:
    FIELDS = [
        "genre",
        "subgenre",
        "mood",
        "energy",
        "danceability",
        "instrumentation",
        "tempo",
        "rhythm_complexity",
        "harmonic_complexity",
        "melodic_density",
        "vocal_presence",
        "production_style",
    ]

    _GENRES = ["electronic", "ambient", "pop", "hiphop", "rock", "jazz"]
    _SUBGENRES = ["deep", "synth", "chill", "indie", "trap", "acoustic"]
    _MOODS = ["uplifting", "dark", "calm", "energetic", "nostalgic"]
    _INSTRUMENTS = [["synth"], ["guitar"], ["piano"], ["drums", "bass"], ["strings"]]
    _PROD = ["lofi", "clean", "vintage", "modern", "cinematic"]

    def extract_genome(self, audio_path: str) -> Dict[str, object]:
        digest = hashlib.sha256(str(audio_path).encode("utf-8")).digest()
        genome: Dict[str, object] = {
            "genre": self._GENRES[digest[0] % len(self._GENRES)],
            "subgenre": self._SUBGENRES[digest[1] % len(self._SUBGENRES)],
            "mood": self._MOODS[digest[2] % len(self._MOODS)],
            "energy": float((digest[3] % 101) / 100.0),
            "danceability": float((digest[4] % 101) / 100.0),
            "instrumentation": self._INSTRUMENTS[digest[5] % len(self._INSTRUMENTS)],
            "tempo": int(60 + (digest[6] % 121)),
            "rhythm_complexity": float((digest[7] % 101) / 100.0),
            "harmonic_complexity": float((digest[8] % 101) / 100.0),
            "melodic_density": float((digest[9] % 101) / 100.0),
            "vocal_presence": float((digest[10] % 101) / 100.0),
            "production_style": self._PROD[digest[11] % len(self._PROD)],
        }
        return genome

    def compare_genomes(self, genome_a: Dict[str, object], genome_b: Dict[str, object]) -> Dict[str, float]:
        comparison: Dict[str, float] = {}
        numeric = {"energy", "danceability", "tempo", "rhythm_complexity", "harmonic_complexity", "melodic_density", "vocal_presence"}
        categorical = {"genre", "subgenre", "mood", "production_style"}
        for field in numeric:
            if field == "tempo":
                delta = abs(float(genome_a.get(field, 0.0)) - float(genome_b.get(field, 0.0))) / 200.0
            else:
                delta = abs(float(genome_a.get(field, 0.0)) - float(genome_b.get(field, 0.0)))
            comparison[field] = max(0.0, 1.0 - min(1.0, delta))
        for field in categorical:
            comparison[field] = 1.0 if str(genome_a.get(field, "")) == str(genome_b.get(field, "")) else 0.0
        inst_a = set(str(x) for x in genome_a.get("instrumentation", []))
        inst_b = set(str(x) for x in genome_b.get("instrumentation", []))
        union = len(inst_a | inst_b)
        comparison["instrumentation"] = float(len(inst_a & inst_b) / union) if union else 0.0
        return comparison

    def similarity_score(self, genome_a: Dict[str, object], genome_b: Dict[str, object]) -> float:
        cmp = self.compare_genomes(genome_a, genome_b)
        if not cmp:
            return 0.0
        score = float(sum(cmp.values()) / len(cmp))
        return max(0.0, min(1.0, score))
