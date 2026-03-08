"""Session-based recommendation model."""

from __future__ import annotations

from collections import defaultdict
from typing import DefaultDict, Dict, List


class SessionModel:
    def __init__(self):
        self.sessions: DefaultDict[str, List[str]] = defaultdict(list)

    def start_session(self, user_id: str) -> Dict[str, object]:
        self.sessions[str(user_id)] = []
        return {"user_id": str(user_id), "session_length": 0}

    def update_session(self, user_id: str, track_id: str) -> Dict[str, object]:
        self.sessions[str(user_id)].append(str(track_id))
        return {"user_id": str(user_id), "session_length": len(self.sessions[str(user_id)])}

    def recommend(self, user_id: str) -> List[str]:
        history = self.sessions.get(str(user_id), [])
        if not history:
            return []
        seen = set()
        ordered = []
        for track_id in reversed(history):
            if track_id not in seen:
                seen.add(track_id)
                ordered.append(track_id)
        return ordered[:10]
