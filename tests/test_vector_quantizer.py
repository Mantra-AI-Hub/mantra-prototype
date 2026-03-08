import numpy as np

from mantra.vector_quantizer import VectorQuantizer


def test_vector_quantizer_train_encode_decode():
    vectors = np.array(
        [
            [1.0, 0.0, 0.1, 0.0],
            [0.9, 0.1, 0.0, 0.1],
            [0.0, 1.0, 0.1, 0.0],
            [0.1, 0.9, 0.0, 0.1],
        ],
        dtype=np.float32,
    )

    q = VectorQuantizer(subspaces=2, codebook_size=2)
    q.train(vectors)

    code = q.encode(np.array([1.0, 0.0, 0.1, 0.0], dtype=np.float32))
    decoded = q.decode(code)

    assert len(code) == 2
    assert decoded.shape == (4,)
    assert np.isfinite(decoded).all()
