"""Synthetic listeners and engagement simulation for Mantra."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
from typing import Dict, List, Sequence, Optional

import numpy as np


@dataclass
class ListenerProfile:
    listener_id: str
    age_group: str
    region: str
    taste_vector: List[float]
    exploration_rate: float
    loyalty: float


@dataclass
class ListeningSession:
    session_id: str
    listener_id: str
    tracks_played: List[str]
    skip_rate: float
    completion_rate: float
    replay_count: int


@dataclass
class TrackEngagement:
    track_id: str
    plays: int
    skips: int
    completions: int
    replays: int
    virality_score: float


class SyntheticListenerEngine:
    def __init__(self, seed: int = 42) -> None:
        self._rng = np.random.default_rng(seed)
        self.listeners: List[ListenerProfile] = []
        self.sessions: List[ListeningSession] = []
        self.track_engagement: Dict[str, TrackEngagement] = {}

    def _cosine_similarity(self, a: Sequence[float], b: Sequence[float]) -> float:
        a_vec = np.asarray(a, dtype=np.float32)
        b_vec = np.asarray(b, dtype=np.float32)
        if a_vec.size == 0 or b_vec.size == 0:
            return 0.0
        norm_prod = np.linalg.norm(a_vec) * np.linalg.norm(b_vec)
        if norm_prod == 0:
            return 0.0
        return float(np.dot(a_vec, b_vec) / norm_prod)

    def generate_listeners(self, n: int = 10000) -> List[ListenerProfile]:
        self.listeners.clear()
        for idx in range(max(1, int(n))):
            vector = self._rng.random(8).tolist()
            listener = ListenerProfile(
                listener_id=f"listener_{idx}",
                age_group=self._rng.choice(["18-24", "25-34", "35-44", "45+"]),
                region=self._rng.choice(["NA", "EU", "APAC", "LATAM"]),
                taste_vector=vector,
                exploration_rate=float(self._rng.uniform(0.1, 0.9)),
                loyalty=float(self._rng.uniform(0.2, 0.95)),
            )
            self.listeners.append(listener)
        return list(self.listeners)

    def simulate_session(self, listener: ListenerProfile, track_pool: List[Dict[str, object]]) -> ListeningSession:
        session_id = f"session_{listener.listener_id}_{len(self.sessions)}"
        tracks = self._rng.choice(track_pool, size=min(5, len(track_pool)), replace=False).tolist()
        tracks_played = []
        skips = 0
        completions = 0
        replays = 0
        for track in tracks:
            track_id = str(track["track_id"])
            similarity = self._cosine_similarity(listener.taste_vector, track.get("vector", []))
            skip_prob = max(0.05, 1.0 - similarity) * (1.0 - listener.loyalty)
            completion_prob = min(1.0, float(track.get("quality_score", 0.6)) * listener.loyalty + listener.loyalty * 0.2)
            replay_prob = min(1.0, float(track.get("emotional_score", 0.5)) * 0.6 + float(track.get("novelty_score", 0.5)) * 0.4)
            skip = self._rng.random() < skip_prob
            if skip:
                skips += 1
            else:
                completions += 1
            replay = self._rng.random() < replay_prob
            if replay:
                replays += 1
            tracks_played.append(track_id)
            self.track_engagement.setdefault(track_id, TrackEngagement(track_id, 0, 0, 0, 0, 0.0))
            engagement = self.track_engagement[track_id]
            engagement.plays += 1
            engagement.skips += int(skip)
            engagement.completions += int(not skip)
            engagement.replays += int(replay)
        skip_rate = skips / max(1, len(tracks))
        completion_rate = completions / max(1, len(tracks))
        session = ListeningSession(
            session_id=session_id,
            listener_id=listener.listener_id,
            tracks_played=tracks_played,
            skip_rate=skip_rate,
            completion_rate=completion_rate,
            replay_count=replays,
        )
        self.sessions.append(session)
        return session

    def simulate_population(
        self, listeners: List[ListenerProfile], track_pool: List[Dict[str, object]], sessions_per_user: int = 5
    ) -> List[ListeningSession]:
        self.sessions.clear()
        for listener in listeners:
            for _ in range(max(1, int(sessions_per_user))):
                self.simulate_session(listener, track_pool)
        return list(self.sessions)

    def compute_engagement(self, sessions: Optional[List[ListeningSession]] = None) -> Dict[str, TrackEngagement]:
        if sessions is None:
            sessions = self.sessions
        engagement_map: Dict[str, TrackEngagement] = {}
        for session in sessions:
            for track_id in session.tracks_played:
                engagement_map.setdefault(track_id, TrackEngagement(track_id, 0, 0, 0, 0, 0.0))
                engagement_map[track_id].plays += 1
        for track_id, engagement in self.track_engagement.items():
            engagement.virality_score = self.compute_virality(engagement)
            engagement_map[track_id] = engagement
        return engagement_map

    def compute_virality(self, track_engagement: TrackEngagement) -> float:
        if track_engagement.plays == 0:
            return 0.0
        completion_rate = track_engagement.completions / track_engagement.plays
        skip_rate = track_engagement.skips / track_engagement.plays
        replay_rate = track_engagement.replays / track_engagement.plays
        hashed = int(hashlib.sha256(track_engagement.track_id.encode("utf-8")).hexdigest(), 16)
        share_probability = float((hashed % 51)) / 100.0
        virality = completion_rate + replay_rate * 0.5 + share_probability * 0.3 - skip_rate * 0.4
        return float(max(0.0, min(1.0, virality)))

    def export_metrics(self) -> Dict[str, object]:
        engagement = self.compute_engagement()
        average_virality = (
            float(np.mean([entry.virality_score for entry in engagement.values()])) if engagement else 0.0
        )
        return {
            "listeners": len(self.listeners),
            "sessions": len(self.sessions),
            "tracks": len(engagement),
            "average_virality": average_virality,
        }

    def simulate_release(self, track_id: str, track_pool: List[Dict[str, object]], sessions: int = 1000) -> TrackEngagement:
        dummy_listener = self.generate_listeners(1)[0]
        track = next((item for item in track_pool if str(item["track_id"]) == track_id), None)
        if track is None:
            raise ValueError("track_id not found")
        pool = [track]
        self.simulate_session(dummy_listener, pool)
        return self.track_engagement.get(track_id, TrackEngagement(track_id, 0, 0, 0, 0, 0.0))

    def simulate_album_release(self, track_ids: List[str], track_pool: List[Dict[str, object]], sessions: int = 1000) -> Dict[str, TrackEngagement]:
        results = {}
        for tid in track_ids:
            results[tid] = self.simulate_release(tid, track_pool, sessions=sessions)
        return results

    def generate_listener_report(self, track_id: str) -> Dict[str, object]:
        engagement = self.track_engagement.get(track_id)
        if not engagement:
            return {"track_id": track_id, "plays": 0, "completion_rate": 0.0, "skip_rate": 0.0, "replay_rate": 0.0, "virality_score": 0.0}
        completion_rate = engagement.completions / max(1, engagement.plays)
        skip_rate = engagement.skips / max(1, engagement.plays)
        replay_rate = engagement.replays / max(1, engagement.plays)
        return {
            "track_id": track_id,
            "plays": engagement.plays,
            "completion_rate": completion_rate,
            "skip_rate": skip_rate,
            "replay_rate": replay_rate,
            "virality_score": engagement.virality_score,
        }


def main() -> int:
    parser = argparse.ArgumentParser(description="Simulate synthetic listeners")
    parser.add_argument("--simulate", action="store_true", help="Run the simulation")
    parser.add_argument("--listeners", type=int, default=10000, help="Number of listeners")
    parser.add_argument("--tracks", type=int, default=50, help="Number of tracks")
    args = parser.parse_args()
    engine = SyntheticListenerEngine(seed=42)
    listeners = engine.generate_listeners(n=args.listeners)
    track_pool = []
    for idx in range(max(1, args.tracks)):
        track_pool.append(
            {
                "track_id": f"track_{idx}",
                "vector": engine._rng.random(8).tolist(),
                "quality_score": engine._rng.uniform(0.5, 0.95),
                "emotional_score": engine._rng.uniform(0.3, 0.9),
                "novelty_score": engine._rng.uniform(0.2, 0.8),
            }
        )
    engine.simulate_population(listeners, track_pool, sessions_per_user=5)
    metrics = engine.export_metrics()
    print("Synthetic Listener Simulation")
    print("============================")
    for key, value in metrics.items():
        print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
