"""Reinforcement-learning style recommender policy."""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Dict, List


class RLRecommender:
    def __init__(self, epsilon: float = 0.1):
        self.epsilon = float(max(0.0, min(1.0, epsilon)))
        self.q_values: Dict[str, Dict[str, float]] = defaultdict(dict)

    def update_reward(self, user_id: str, track_id: str, reward: float) -> None:
        uid = str(user_id)
        tid = str(track_id)
        prev = float(self.q_values[uid].get(tid, 0.0))
        self.q_values[uid][tid] = 0.8 * prev + 0.2 * float(reward)

    def choose_action(self, user_id: str, candidate_tracks: List[Dict[str, object]]) -> Dict[str, object] | None:
        if not candidate_tracks:
            return None

        if random.random() < self.epsilon:
            return random.choice(candidate_tracks)

        uid = str(user_id)
        ranked = []
        for item in candidate_tracks:
            tid = str(item.get("track_id"))
            base = float(item.get("rank_score", item.get("score", 0.0)))
            bonus = float(self.q_values[uid].get(tid, 0.0))
            ranked.append((base + bonus, item))
        ranked.sort(key=lambda x: x[0], reverse=True)
        return ranked[0][1]
