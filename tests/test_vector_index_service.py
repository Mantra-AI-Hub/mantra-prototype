from uuid import uuid4

from mantra.vector_index_service import VectorIndexService


def test_vector_index_service_build_add_search_save_load():
    base = f"test_vector_service_{uuid4().hex}"
    svc = VectorIndexService(index_path=base)
    svc.build_index(4)
    svc.add_vector("a", [1.0, 0.0, 0.0, 0.0])
    svc.add_vector("b", [0.0, 1.0, 0.0, 0.0])

    results = svc.search([0.9, 0.1, 0.0, 0.0], 1)
    assert results
    assert results[0][0] == "a"

    svc.save_index()
    loaded = VectorIndexService(index_path=base)
    loaded.load_index()
    results2 = loaded.search([0.0, 1.0, 0.0, 0.0], 1)
    assert results2
    assert results2[0][0] == "b"
