from mantra.ai_producer_engine import AIProducerEngine


def test_ai_producer_engine_pipeline():
    engine = AIProducerEngine()
    melody = engine.generate_melody(seed=60, bars=1)
    rhythm = engine.generate_rhythm(bars=1)
    mix = engine.mix_stems([[float(n) / 127.0 for n in melody], rhythm])
    mastered = engine.simulate_mastering_pipeline(mix)
    assert melody
    assert rhythm
    assert len(mastered) == len(mix)
    genome_track = engine.generate_track_from_genome({"tempo": 120, "energy": 0.7, "melodic_density": 0.5})
    assert "mastered" in genome_track
