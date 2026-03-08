"""Generation-facing API for AIProducerEngine."""

from __future__ import annotations

from typing import Dict, List, Optional

from mantra.ai_producer_engine import AIProducerEngine as BaseAIProducerEngine


class AIProducerEngine(BaseAIProducerEngine):
    def produce_track(self, genre: Optional[str] = None, mood: Optional[str] = None, tempo: Optional[int] = None, genome: Optional[Dict[str, object]] = None) -> Dict[str, object]:
        context_genome = genome or {
            "tempo": max(60, min(180, (tempo or 120))),
            "energy": 0.5,
            "melodic_density": 0.5,
        }
        return self.generate_track_from_genome(context_genome)


__all__ = ["AIProducerEngine"]
