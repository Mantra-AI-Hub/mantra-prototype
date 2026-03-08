from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest

from mantra.vector_index.faiss_index import VectorIndex


def test_vector_index_add_and_search_top_match():
    index = VectorIndex(dimension=4)
    index.add("track_a", np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
    index.add("track_b", np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32))
    index.add("track_c", np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32))

    results = index.search(np.array([0.05, 0.95, 0.0, 0.0], dtype=np.float32), top_k=2)

    assert len(results) == 2
    assert results[0][0] == "track_b"
    assert results[0][1] >= results[1][1]


def test_vector_index_save_and_load_roundtrip():
    base = Path(f"test_vector_index_{uuid4().hex}")

    index = VectorIndex(dimension=3)
    index.add("song_1", [1.0, 0.0, 0.0])
    index.add("song_2", [0.0, 1.0, 0.0])
    index.save(str(base))

    try:
        loaded = VectorIndex.load(str(base))
        results = loaded.search([0.0, 1.0, 0.0], top_k=1)
        assert results
        assert results[0][0] == "song_2"
    finally:
        for suffix in (".faiss", ".meta.json", ".npz"):
            base.with_suffix(base.suffix + suffix).unlink(missing_ok=True)


def test_vector_index_empty_search_and_dimension_validation():
    index = VectorIndex(dimension=2)
    assert index.search([1.0, 0.0], top_k=5) == []

    index.add("x", [1.0, 0.0])
    with pytest.raises(ValueError):
        index.search([1.0, 0.0, 0.0], top_k=1)
