"""Service wrapper for AI DJ session generation."""

from __future__ import annotations

from mantra.ai_dj import AIDJ


class AIDJService:
    def __init__(self):
        self.dj = AIDJ()

    def create_session(self, user_id: str, recommendations, persona: str = "classic"):
        return self.dj.generate_session(user_id=user_id, recommendations=recommendations, persona=persona)

