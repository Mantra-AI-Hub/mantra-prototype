from uuid import uuid4

from mantra.feature_store import FeatureStore


def test_feature_store_user_and_track_interfaces():
    db_path = f"test_feature_store_extended_{uuid4().hex}.db"
    store = FeatureStore(db_path=db_path)

    store.store_user_features("u1", {"embedding": [1.0, 0.0], "taste": "ambient"})
    store.store_track_features("t1", {"embedding": [0.8, 0.2], "genre": "ambient"})

    u = store.get_user_features("u1")
    t = store.get_track_features("t1")

    assert u is not None and u["taste"] == "ambient"
    assert t is not None and t["genre"] == "ambient"
