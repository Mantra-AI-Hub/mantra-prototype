from uuid import uuid4

from mantra.feature_store import FeatureStore


def test_feature_store_put_get_batch():
    db_path = f"test_feature_store_{uuid4().hex}.db"
    store = FeatureStore(db_path=db_path)

    store.put("t1", {"popularity": 0.2, "genre": "ambient"})
    store.put("t2", {"popularity": 0.5, "genre": "rock"})

    one = store.get("t1")
    assert one is not None
    assert one["genre"] == "ambient"

    batch = store.batch_get(["t1", "t2", "t3"])
    assert batch[0]["genre"] == "ambient"
    assert batch[1]["genre"] == "rock"
    assert batch[2] is None
