from mantra.music_generation_pipeline import MusicGenerationPipeline


def test_music_generation_pipeline_generate_and_style_transfer():
    pipeline = MusicGenerationPipeline()
    generated = pipeline.generate_music(prompt="deep ambient pulse", seconds=1, sample_rate=2000)
    assert generated["samples"]
    assert generated["sample_rate"] == 2000

    styled = pipeline.style_transfer(track="t1", style="lofi")
    assert styled["output_track"] == "t1__lofi"

    loops = pipeline.generate_loops(count=3)
    assert len(loops) == 3

