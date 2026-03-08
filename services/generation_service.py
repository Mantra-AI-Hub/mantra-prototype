"""Service wrapper for music generation pipeline."""

from __future__ import annotations

from mantra.music_generation_pipeline import MusicGenerationPipeline


class GenerationService:
    def __init__(self):
        self.pipeline = MusicGenerationPipeline()

    def generate(self, prompt: str, seconds: int = 5):
        return self.pipeline.generate_music(prompt=prompt, seconds=seconds)

    def style_transfer(self, track: str, style: str):
        return self.pipeline.style_transfer(track=track, style=style)

