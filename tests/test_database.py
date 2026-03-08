from pathlib import Path
from uuid import uuid4

import numpy as np

from mantra.database.track_store import TrackStore
from mantra.vector_index.faiss_index import VectorIndex


def test_track_store_add_get_list_delete_cycle():
    db_path = f"test_tracks_{uuid4().hex}.db"
    store = TrackStore(db_path=db_path)

    metadata = {
        "track_id": "track_1",
        "filename": "song.wav",
        "duration": 12.5,
        "embedding_path": "vector_store",
        "fingerprint_hash_count": 42,
        "created_at": "2026-03-07T00:00:00+00:00",
    }

    store.add_track(metadata)
    record = store.get_track("track_1")
    assert record is not None
    assert record["filename"] == "song.wav"

    listed = store.list_tracks()
    assert listed
    assert listed[0]["track_id"] == "track_1"

    store.delete_track("track_1")
    assert store.get_track("track_1") is None

def test_track_store_fingerprint_roundtrip_and_vector_index_save_load():
    run_id = uuid4().hex
    db_path = f"test_tracks_{run_id}.db"
    vector_base = f"test_vector_store_{run_id}"

    store = TrackStore(db_path=db_path)
    store.add_track(
        {
            "track_id": "track_2",
            "filename": "song2.wav",
            "duration": 5.0,
            "embedding_path": vector_base,
            "fingerprint_hash_count": 2,
            "created_at": "2026-03-07T00:00:00+00:00",
        }
    )

    fingerprint = [("10:20:3", 5), ("10:22:4", 8)]
    store.upsert_fingerprint("track_2", fingerprint)

    assert store.get_fingerprint("track_2") == fingerprint
    all_fps = store.list_fingerprints()
    assert all_fps["track_2"] == fingerprint

    index = VectorIndex(dimension=4)
    index.add("track_2", np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
    index.save(vector_base)

    loaded = VectorIndex.load(vector_base)
    results = loaded.search(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), top_k=1)
    assert results
    assert results[0][0] == "track_2"

    for suffix in (".faiss", ".meta.json", ".npz"):
        Path(vector_base + suffix).unlink(missing_ok=True)
