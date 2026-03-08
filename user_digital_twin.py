"""Digital twin simulation for listener behavior."""

from __future__ import annotations

import logging
from typing import Dict, List


class UserDigitalTwin:
    def __init__(self) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self.user_profiles: Dict[str, Dict[str, float]] = {}
        self.metrics: Dict[str, float | int] = {
            "users": 0,
            "sessions_simulated": 0,
            "taste_updates": 0,
        }

    def simulate_user_taste(self, user_id: str, taste_vector: Dict[str, float]) -> Dict[str, float]:
        normalized = {str(k): max(0.0, float(v)) for k, v in taste_vector.items()}
        total = sum(normalized.values()) or 1.0
        normalized = {k: float(v / total) for k, v in normalized.items()}
        self.user_profiles[str(user_id)] = normalized
        self.metrics["users"] = len(self.user_profiles)
        self.metrics["taste_updates"] = int(self.metrics["taste_updates"]) + 1
        return normalized

    def simulate_listening_sessions(self, user_id: str, candidates: List[Dict[str, object]], steps: int = 10) -> List[Dict[str, object]]:
        profile = self.user_profiles.get(str(user_id), {})
        ordered = sorted(
            candidates,
            key=lambda item: float(profile.get(str(item.get("genre", "")), 0.0)),
            reverse=True,
        )
        session = ordered[: max(1, int(steps))]
        self.metrics["sessions_simulated"] = int(self.metrics["sessions_simulated"]) + 1
        self.logger.info("Simulated session for %s with %d tracks", user_id, len(session))
        return session

    def evaluate_recommendation_changes(
        self, user_id: str, baseline: List[Dict[str, object]], candidate: List[Dict[str, object]]
    ) -> Dict[str, float]:
        profile = self.user_profiles.get(str(user_id), {})

        def _score(items: List[Dict[str, object]]) -> float:
            if not items:
                return 0.0
            return float(sum(profile.get(str(it.get("genre", "")), 0.0) for it in items) / len(items))

        baseline_score = _score(baseline)
        candidate_score = _score(candidate)
        return {
            "baseline_score": baseline_score,
            "candidate_score": candidate_score,
            "delta": candidate_score - baseline_score,
        }

    def metrics_snapshot(self) -> Dict[str, float | int]:
        return dict(self.metrics)
