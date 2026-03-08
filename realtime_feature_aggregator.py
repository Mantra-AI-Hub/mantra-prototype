"""Rolling real-time feature aggregation from event stream."""

from __future__ import annotations

from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone
from typing import Deque, Dict, Iterable, List


class RealtimeFeatureAggregator:
    def __init__(self, window_minutes: int = 60):
        self.window_minutes = max(1, int(window_minutes))
        self.track_events: Dict[str, Deque[Dict[str, object]]] = defaultdict(deque)

    def _parse_ts(self, value: object) -> datetime:
        if isinstance(value, datetime):
            ts = value
        else:
            ts = datetime.fromisoformat(str(value))
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return ts

    def _prune(self, track_id: str) -> None:
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(minutes=self.window_minutes)
        queue = self.track_events.get(track_id, deque())
        while queue:
            try:
                ts = self._parse_ts(queue[0].get("timestamp"))
            except Exception:
                queue.popleft()
                continue
            if ts < cutoff:
                queue.popleft()
            else:
                break
        self.track_events[track_id] = queue

    def update_from_event(self, event: Dict[str, object]) -> None:
        track_id = str(event.get("track_id") or "")
        if not track_id:
            return
        payload = dict(event)
        payload.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
        self.track_events[track_id].append(payload)
        self._prune(track_id)

    def compute_features(self, track_id: str) -> Dict[str, float]:
        tid = str(track_id)
        self._prune(tid)
        events = list(self.track_events.get(tid, []))
        if not events:
            return {"play_velocity": 0.0, "skip_rate": 0.0, "engagement_score": 0.0}
        plays = sum(1 for e in events if str(e.get("event")) in {"play", "click", "like"})
        skips = sum(1 for e in events if str(e.get("event")) == "skip")
        likes = sum(1 for e in events if str(e.get("event")) == "like")
        velocity = float(plays) / float(self.window_minutes)
        skip_rate = float(skips) / float(max(1, plays + skips))
        engagement = float(likes + plays) / float(max(1, len(events)))
        return {"play_velocity": velocity, "skip_rate": skip_rate, "engagement_score": engagement}

    def snapshot(self) -> Dict[str, Dict[str, float]]:
        return {track_id: self.compute_features(track_id) for track_id in list(self.track_events.keys())}


