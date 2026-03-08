"""Contextual bandit recommender with multiple exploration policies."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Dict, Iterable, List


class BanditRecommender:
    def __init__(self, epsilon: float = 0.1, policy: str = "thompson"):
        self.epsilon = max(0.0, min(1.0, float(epsilon)))
        self.policy = str(policy).lower()
        self.stats: Dict[str, Dict[str, float]] = defaultdict(lambda: {"n": 0.0, "reward": 0.0, "reward_sq": 0.0})
        self.total_steps = 0

    def update_reward(self, user_id: str, track_id: str, reward: float) -> None:
        key = f"{user_id}:{track_id}"
        row = self.stats[key]
        row["n"] += 1.0
        row["reward"] += float(reward)
        row["reward_sq"] += float(reward) ** 2
        self.total_steps += 1

    def _mean(self, user_id: str, track_id: str) -> float:
        row = self.stats[f"{user_id}:{track_id}"]
        n = row["n"]
        return float(row["reward"] / n) if n > 0 else 0.0

    def _thompson_score(self, user_id: str, track_id: str) -> float:
        row = self.stats[f"{user_id}:{track_id}"]
        alpha = 1.0 + max(0.0, row["reward"])
        beta = 1.0 + max(0.0, row["n"] - row["reward"])
        return random.betavariate(alpha, beta)

    def _ucb_score(self, user_id: str, track_id: str) -> float:
        row = self.stats[f"{user_id}:{track_id}"]
        n = max(1.0, row["n"])
        mean = self._mean(user_id, track_id)
        bonus = math.sqrt(2.0 * math.log(max(2.0, float(self.total_steps) + 1.0)) / n)
        return float(mean + bonus)

    def score_candidates(self, user_id: str, candidates: Iterable[Dict[str, object]]) -> List[Dict[str, object]]:
        items = [dict(item) for item in candidates]
        if not items:
            return items

        uid = str(user_id)
        if self.policy == "epsilon-greedy":
            for item in items:
                tid = str(item.get("track_id"))
                exploit = self._mean(uid, tid)
                explore = random.random()
                item["bandit_score"] = float(exploit if explore > self.epsilon else explore)
            return items

        if self.policy == "ucb":
            for item in items:
                tid = str(item.get("track_id"))
                item["bandit_score"] = float(self._ucb_score(uid, tid))
            return items

        # default: Thompson Sampling
        for item in items:
            tid = str(item.get("track_id"))
            item["bandit_score"] = float(self._thompson_score(uid, tid))
        return items


