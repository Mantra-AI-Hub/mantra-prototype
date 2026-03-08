from mantra.intelligence.music_genome_engine import MusicGenomeEngine
from mantra.intelligence.music_genome_store import MusicGenomeStore


def test_music_genome_store_roundtrip_and_search():
    engine = MusicGenomeEngine()
    store = MusicGenomeStore(genome_engine=engine)
    genome_a = engine.extract_genome("alpha.wav")
    genome_b = engine.extract_genome("beta.wav")
    store.store_genome("a", genome_a)
    store.store_genome("b", genome_b)
    loaded = store.get_genome("a")
    assert loaded is not None
    results = store.search_similar(genome_a, top_k=2)
    assert results
    assert results[0]["track_id"] in {"a", "b"}
