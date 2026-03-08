from uuid import uuid4

from mantra.fingerprint_index import FingerprintIndex


def test_fingerprint_index_add_query_and_persist():
    index = FingerprintIndex()
    fp_a = [("1:2:3", 10), ("2:3:4", 12)]
    fp_b = [("5:6:3", 10)]

    index.add("track_a", fp_a)
    index.add("track_b", fp_b)

    results = index.query([("1:2:3", 11), ("2:3:4", 13)])
    assert results
    assert results[0]["track_id"] == "track_a"

    path = f"test_fp_index_{uuid4().hex}.json"
    index.save(path)
    loaded = FingerprintIndex.load(path)
    loaded_results = loaded.query([("1:2:3", 11)])
    assert loaded_results
    assert loaded_results[0]["track_id"] == "track_a"
