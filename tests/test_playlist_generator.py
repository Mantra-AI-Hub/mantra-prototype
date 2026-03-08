from uuid import uuid4

import mantra.interfaces.api.api_server as api_server
from mantra.database.track_store import TrackStore
from mantra.recommendation_engine import RecommendationEngine


def test_playlist_generator_endpoint():
    db_path = f"test_playlist_{uuid4().hex}.db"
    store = TrackStore(db_path=db_path)
    store.add_track(
        {
            "track_id": "seed",
            "filename": "seed.wav",
            "duration": 1.0,
            "embedding_path": "v",
            "fingerprint_hash_count": 10,
            "created_at": "2026-01-01T00:00:00+00:00",
            "artist": "a",
            "album": "x",
            "genre": "ambient",
            "tags": ["calm"],
            "year": 2020,
        }
    )
    store.add_track(
        {
            "track_id": "n1",
            "filename": "n1.wav",
            "duration": 1.0,
            "embedding_path": "v",
            "fingerprint_hash_count": 11,
            "created_at": "2026-01-01T00:00:00+00:00",
            "artist": "b",
            "album": "y",
            "genre": "ambient",
            "tags": ["calm"],
            "year": 2021,
        }
    )

    api_server._track_store = store
    api_server._recommendation_engine = RecommendationEngine(track_store=store)
    api_server._recommendation_engine.build_similarity_graph()

    resp = api_server.playlist_generate(seed_track_id="seed", length=2)
    assert resp["playlist"]
    assert resp["playlist"][0]["track_id"] == "seed"
