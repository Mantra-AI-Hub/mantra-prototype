from uuid import uuid4

from mantra.database.track_store import TrackStore
from mantra.recommendation_engine import RecommendationEngine


def test_recommendation_engine_graph_and_recommend():
    db_path = f"test_reco_{uuid4().hex}.db"
    store = TrackStore(db_path=db_path)

    store.add_track({
        "track_id": "t1",
        "filename": "a.wav",
        "duration": 1.0,
        "embedding_path": "v",
        "fingerprint_hash_count": 10,
        "created_at": "2026-01-01T00:00:00+00:00",
        "artist": "artist_a",
        "genre": "ambient",
        "album": "x",
        "tags": ["calm"],
        "year": 2021,
    })
    store.add_track({
        "track_id": "t2",
        "filename": "b.wav",
        "duration": 1.0,
        "embedding_path": "v",
        "fingerprint_hash_count": 10,
        "created_at": "2026-01-01T00:00:00+00:00",
        "artist": "artist_b",
        "genre": "ambient",
        "album": "y",
        "tags": ["calm"],
        "year": 2022,
    })

    engine = RecommendationEngine(track_store=store)
    graph = engine.build_similarity_graph()
    assert "t1" in graph

    recs = engine.recommend("t1", k=5)
    assert recs
    assert recs[0][0] == "t2"
