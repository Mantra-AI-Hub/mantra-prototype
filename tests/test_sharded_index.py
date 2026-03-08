from uuid import uuid4

from mantra.sharded_index import ShardedVectorIndex


def test_sharded_index_add_search_save_load():
    base = f"test_sharded_{uuid4().hex}"
    index = ShardedVectorIndex(dimension=4, shard_count=3, strategy="hash")
    index.add([1.0, 0.0, 0.0, 0.0], "a")
    index.add([0.0, 1.0, 0.0, 0.0], "b")
    index.add([0.0, 0.0, 1.0, 0.0], "c")

    results = index.search([0.9, 0.1, 0.0, 0.0], k=2)
    assert results
    assert results[0][0] == "a"

    index.save(base)
    loaded = ShardedVectorIndex.load(base)
    loaded_results = loaded.search([0.0, 1.0, 0.0, 0.0], k=1)
    assert loaded_results
    assert loaded_results[0][0] == "b"
