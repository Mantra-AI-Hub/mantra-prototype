import numpy as np
from uuid import uuid4

from mantra.vector_index import VectorIndex


def test_vector_index_interface_add_search_save_load():
    path = f"test_large_index_{uuid4().hex}"
    index = VectorIndex(dimension=4)
    index.add([1.0, 0.0, 0.0, 0.0], "x")
    index.add([0.0, 1.0, 0.0, 0.0], "y")

    results = index.search([0.9, 0.1, 0.0, 0.0], k=1)
    assert results
    assert results[0][0] == "x"

    index.save(path)
    loaded = VectorIndex.load(path)
    loaded_results = loaded.search(np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32), k=1)
    assert loaded_results
    assert loaded_results[0][0] == "y"
