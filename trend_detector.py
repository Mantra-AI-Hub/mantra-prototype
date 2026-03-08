"""Trend and virality detection from event streams."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Dict, List


class TrendDetector:
    def __init__(self):
        self.events: List[Dict[str, object]] = []

    def ingest_event(self, event: Dict[str, object]) -> None:
        e = dict(event)
        e.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
        self.events.append(e)

    def _within_window(self, hours: int) -> List[Dict[str, object]]:
        now = datetime.now(timezone.utc)
        start = now - timedelta(hours=max(1, int(hours)))
        out = []
        for e in self.events:
            try:
                ts = datetime.fromisoformat(str(e.get("timestamp")))
            except Exception:
                continue
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            if ts >= start:
                out.append(e)
        return out

    def detect_trending_tracks(self, events: List[Dict[str, object]] | None = None, top_k: int = 20) -> List[Dict[str, object]]:
        source = events if events is not None else self.events
        counts: Dict[str, int] = defaultdict(int)
        for e in source:
            track = str(e.get("track_id") or "")
            if not track:
                continue
            counts[track] += 1
        ranked = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        return [{"track_id": tid, "count": c} for tid, c in ranked[: max(0, int(top_k))]]

    def compute_velocity(self, track_id: str) -> float:
        recent = [e for e in self._within_window(1) if str(e.get("track_id")) == str(track_id)]
        baseline = [e for e in self._within_window(24) if str(e.get("track_id")) == str(track_id)]
        r = float(len(recent))
        b = float(len(baseline)) / 24.0 if baseline else 0.0
        return r - b

    def detect_viral_tracks(self, window_hours: int = 24, top_k: int = 20) -> List[Dict[str, object]]:
        window_events = self._within_window(window_hours)
        tracks = {str(e.get("track_id")) for e in window_events if e.get("track_id")}
        scored = []
        for tid in tracks:
            velocity = self.compute_velocity(tid)
            scored.append({"track_id": tid, "velocity": float(velocity)})
        scored.sort(key=lambda x: x["velocity"], reverse=True)
        return scored[: max(0, int(top_k))]
