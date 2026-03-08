import numpy as np

from mantra.ann_router import ANNRouter


def test_ann_router_train_and_route():
    vectors = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.9, 0.1],
        ],
        dtype=np.float32,
    )

    router = ANNRouter(clusters=2)
    router.train(vectors)
    shard_ids = router.route(np.array([1.0, 0.0, 0.0], dtype=np.float32), top_shards=1)

    assert len(shard_ids) == 1
    assert isinstance(shard_ids[0], int)
