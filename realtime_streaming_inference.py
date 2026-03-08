"""Real-time streaming inference engine subscribed to event stream."""

from __future__ import annotations

from collections import defaultdict
from typing import Callable, Dict, List


class RealtimeStreamingInference:
    def __init__(self, event_stream, recompute_fn: Callable[[str], List[Dict[str, object]]] | None = None):
        self.event_stream = event_stream
        self.recompute_fn = recompute_fn
        self.active = False
        self.user_session: Dict[str, List[str]] = defaultdict(list)
        self.latest_recommendations: Dict[str, List[Dict[str, object]]] = {}

    def _on_event(self, event: Dict[str, object]) -> None:
        user_id = str(event.get("user_id") or "")
        track_id = str(event.get("track_id") or "")
        if not user_id:
            return
        if track_id:
            self.user_session[user_id].append(track_id)
        if self.recompute_fn is not None:
            self.latest_recommendations[user_id] = self.recompute_fn(user_id)

    def start(self) -> None:
        if self.active:
            return
        self.event_stream.register_handler(self._on_event)
        self.active = True

    def status(self) -> Dict[str, object]:
        return {
            "active": self.active,
            "users": len(self.user_session),
            "sessions": {uid: len(hist) for uid, hist in self.user_session.items()},
        }


