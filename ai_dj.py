"""AI DJ session builder for continuous recommendation + narration."""

from __future__ import annotations

from typing import Dict, List


class AIDJ:
    def __init__(self):
        self.personas = {
            "chill": "Smooth host with mellow transitions",
            "energy": "High-energy host with punchy intros",
            "classic": "Warm radio-style curator",
        }

    def persona_prompt(self, persona: str) -> str:
        p = str(persona or "classic").lower()
        return self.personas.get(p, self.personas["classic"])

    def generate_session(self, user_id: str, recommendations: List[Dict[str, object]], persona: str = "classic") -> Dict[str, object]:
        prompt = self.persona_prompt(persona)
        segments = []
        for idx, item in enumerate(recommendations):
            track_id = str(item.get("track_id"))
            score = float(item.get("score", item.get("rank_score", 0.0)))
            narration = f"{prompt}: Now playing {track_id} with confidence {score:.2f}."
            segments.append({"position": idx + 1, "track_id": track_id, "score": score, "narration": narration})
        return {"user_id": user_id, "persona": persona, "segments": segments}


