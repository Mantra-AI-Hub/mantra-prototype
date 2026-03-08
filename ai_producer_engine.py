"""AI producer engine for melody, rhythm, mixing, and mastering simulation."""

from __future__ import annotations

import logging
from typing import Dict, List


class AIProducerEngine:
    def __init__(self) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self.metrics: Dict[str, float | int] = {"melodies": 0, "rhythms": 0, "mixes": 0, "masters": 0}

    def generate_melody(self, seed: int = 60, bars: int = 2) -> List[int]:
        notes = [int(seed + ((i * 3) % 7)) for i in range(max(1, bars * 8))]
        self.metrics["melodies"] = int(self.metrics["melodies"]) + 1
        return notes

    def generate_rhythm(self, bars: int = 2) -> List[float]:
        pattern = [1.0 if i % 4 in {0, 2} else 0.5 for i in range(max(1, bars * 8))]
        self.metrics["rhythms"] = int(self.metrics["rhythms"]) + 1
        return pattern

    def mix_stems(self, stems: List[List[float]]) -> List[float]:
        if not stems:
            return []
        length = min(len(stem) for stem in stems)
        mixed = [float(sum(stem[i] for stem in stems) / len(stems)) for i in range(length)]
        self.metrics["mixes"] = int(self.metrics["mixes"]) + 1
        return mixed

    def simulate_mastering_pipeline(self, signal: List[float]) -> List[float]:
        if not signal:
            return []
        peak = max(abs(v) for v in signal) or 1.0
        mastered = [float(max(-1.0, min(1.0, (v / peak) * 0.92))) for v in signal]
        self.metrics["masters"] = int(self.metrics["masters"]) + 1
        self.logger.info("Mastered signal with %d samples", len(mastered))
        return mastered

    def metrics_snapshot(self) -> Dict[str, float | int]:
        return dict(self.metrics)

    def generate_track_from_genome(self, genome: Dict[str, object]) -> Dict[str, object]:
        tempo = int(genome.get("tempo", 120))
        bars = max(1, min(8, tempo // 30))
        seed = 48 + int(float(genome.get("melodic_density", 0.5)) * 24)
        melody = self.generate_melody(seed=seed, bars=bars)
        rhythm = self.generate_rhythm(bars=bars)
        energy = float(genome.get("energy", 0.5))
        stem_melody = [float(note) / 127.0 * (0.5 + energy) for note in melody]
        stem_rhythm = [float(value) * (0.4 + energy) for value in rhythm]
        mix = self.mix_stems([stem_melody, stem_rhythm])
        mastered = self.simulate_mastering_pipeline(mix)
        return {"genome": dict(genome), "melody": melody, "rhythm": rhythm, "mix": mix, "mastered": mastered}
