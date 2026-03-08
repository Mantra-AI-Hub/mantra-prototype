"""Adaptive style engine for detecting and evolving musical styles."""

from __future__ import annotations

import logging
from collections import Counter
from typing import Dict, List


class AdaptiveStyleEngine:
    def __init__(self) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self.style_scores: Dict[str, float] = {}
        self.metrics: Dict[str, float | int] = {"detections": 0, "evolutions": 0}

    def detect_new_musical_styles(self, tracks: List[Dict[str, object]]) -> List[str]:
        counter = Counter(str(track.get("style", "unknown")) for track in tracks)
        emerging = [style for style, count in counter.items() if count >= 2]
        self.metrics["detections"] = int(self.metrics["detections"]) + 1
        return emerging

    def evolve_generation_models(self, detected_styles: List[str]) -> Dict[str, float]:
        for style in detected_styles:
            self.style_scores[style] = float(self.style_scores.get(style, 1.0) * 1.05)
        self.metrics["evolutions"] = int(self.metrics["evolutions"]) + 1
        self.logger.info("Evolved model weights for %d styles", len(detected_styles))
        return dict(self.style_scores)

    def metrics_snapshot(self) -> Dict[str, float | int]:
        return {**self.metrics, "tracked_styles": len(self.style_scores)}
