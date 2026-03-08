from mantra.generation_orchestrator import GenerationOrchestrator
from mantra.music_generation_pipeline import MusicGenerationPipeline


def test_generation_orchestrator_generate_and_remix():
    orchestrator = GenerationOrchestrator(MusicGenerationPipeline())
    generated = orchestrator.generate_advanced(prompt="deep house lead", mode="audiogen", seconds=1)
    assert generated["mode"] == "audiogen"
    remix = orchestrator.remix(track="t1", style="club")
    assert remix["output_track"] == "t1__remix-club"

