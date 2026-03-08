"""Collaborative AI band where multiple agents co-create tracks."""

from __future__ import annotations

import logging
from typing import Dict, List


class CollaborativeAIBand:
    def __init__(self, agents: List[str] | None = None) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self.agents = agents or ["drummer", "bassist", "lead", "arranger"]
        self.metrics: Dict[str, float | int] = {"sessions": 0, "agents": len(self.agents)}

    def create_track_collaboratively(self, prompt: str, rounds: int = 2) -> Dict[str, object]:
        contributions: List[Dict[str, str]] = []
        for round_idx in range(max(1, int(rounds))):
            for agent in self.agents:
                contributions.append(
                    {"round": str(round_idx), "agent": agent, "idea": f"{agent}:{prompt}:variation_{round_idx}"}
                )
        self.metrics["sessions"] = int(self.metrics["sessions"]) + 1
        self.logger.info("Collaborative session produced %d contributions", len(contributions))
        return {"prompt": prompt, "contributions": contributions}

    def metrics_snapshot(self) -> Dict[str, float | int]:
        return dict(self.metrics)
