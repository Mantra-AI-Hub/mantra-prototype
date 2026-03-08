"""Ecosystem-level simulation for users, artists, playlists, and trends."""

from __future__ import annotations

import logging
import random
from typing import Dict, List


class EcosystemSimulator:
    def __init__(self, seed: int = 7) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self._rng = random.Random(seed)
        self.state: Dict[str, List[Dict[str, object]]] = {"users": [], "artists": [], "playlists": []}
        self.metrics: Dict[str, float | int] = {
            "simulation_runs": 0,
            "trend_emergence_score": 0.0,
        }

    def simulate_users_artists_playlists(
        self, n_users: int = 50, n_artists: int = 12, n_playlists: int = 20
    ) -> Dict[str, List[Dict[str, object]]]:
        users = [{"user_id": f"user_{i}", "activity": self._rng.random()} for i in range(max(1, int(n_users)))]
        artists = [{"artist_id": f"artist_{i}", "momentum": self._rng.random()} for i in range(max(1, int(n_artists)))]
        playlists = []
        for idx in range(max(1, int(n_playlists))):
            sample_artists = self._rng.sample(artists, k=min(3, len(artists)))
            playlists.append({"playlist_id": f"playlist_{idx}", "artists": [a["artist_id"] for a in sample_artists]})
        self.state = {"users": users, "artists": artists, "playlists": playlists}
        self.metrics["simulation_runs"] = int(self.metrics["simulation_runs"]) + 1
        return self.state

    def simulate_trend_emergence(self) -> Dict[str, object]:
        artists = self.state.get("artists", [])
        ranked = sorted(artists, key=lambda a: float(a.get("momentum", 0.0)), reverse=True)
        leaders = ranked[:3]
        score = float(sum(float(a.get("momentum", 0.0)) for a in leaders))
        self.metrics["trend_emergence_score"] = score
        return {"emerging_artists": leaders, "trend_score": score}

    def metrics_snapshot(self) -> Dict[str, float | int]:
        return dict(self.metrics)
