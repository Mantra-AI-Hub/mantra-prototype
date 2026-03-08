import numpy as np

from mantra.coarse_index import CoarseIndex


def test_coarse_index_training_assignment_candidates():
    vectors = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.9, 0.1],
        ],
        dtype=np.float32,
    )

    index = CoarseIndex(clusters=2)
    index.train(vectors)

    cid = index.assign(np.array([1.0, 0.0, 0.0], dtype=np.float32))
    candidates = index.get_candidate_clusters(np.array([1.0, 0.0, 0.0], dtype=np.float32), n=2)

    assert cid in candidates
    assert len(candidates) >= 1
