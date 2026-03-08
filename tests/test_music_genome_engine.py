from mantra.intelligence.music_genome_engine import MusicGenomeEngine


def test_music_genome_extract_and_similarity():
    engine = MusicGenomeEngine()
    genome_a = engine.extract_genome("track_a.wav")
    genome_b = engine.extract_genome("track_b.wav")
    assert "genre" in genome_a
    assert "tempo" in genome_a
    score = engine.similarity_score(genome_a, genome_b)
    assert 0.0 <= score <= 1.0
    compare = engine.compare_genomes(genome_a, genome_b)
    assert "energy" in compare
