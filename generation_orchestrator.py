"""Orchestration for generation, remix, and style transfer."""

from __future__ import annotations

from typing import Dict


class GenerationOrchestrator:
    def __init__(self, generation_pipeline):
        self.pipeline = generation_pipeline

    def generate_advanced(self, prompt: str, mode: str = "musicgen", seconds: int = 5) -> Dict[str, object]:
        base = self.pipeline.generate_music(prompt=prompt, seconds=seconds)
        base["mode"] = mode
        return base

    def remix(self, track: str, style: str = "electronic") -> Dict[str, object]:
        return self.pipeline.style_transfer(track=track, style=f"remix-{style}")

