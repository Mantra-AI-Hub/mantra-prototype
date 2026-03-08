from mantra.artist_intelligence import ArtistIntelligence


def test_artist_intelligence_similarity_and_recommend():
    engine = ArtistIntelligence()
    tracks = [
        {"track_id": "t1", "artist": "a1", "genre": "ambient", "year": 2020},
        {"track_id": "t2", "artist": "a2", "genre": "ambient", "year": 2021},
        {"track_id": "t3", "artist": "a3", "genre": "rock", "year": 2000},
    ]
    graph = engine.build_artist_graph(tracks)
    assert "a1" in graph or "a2" in graph

    sim = engine.compute_artist_similarity("a1", top_k=5)
    assert isinstance(sim, list)

    engine.record_user_artist("u1", "a1")
    recs = engine.recommend_artists("u1", k=5)
    assert isinstance(recs, list)

