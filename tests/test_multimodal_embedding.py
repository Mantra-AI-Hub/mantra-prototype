import numpy as np

from mantra.multimodal_embedding import fuse_embeddings


def test_multimodal_fusion_output_shape_and_norm():
    audio = np.ones(87, dtype=np.float32)
    text = np.ones(128, dtype=np.float32) * 0.5
    meta = np.array([0.1, 0.2, 0.3], dtype=np.float32)

    fused = fuse_embeddings(audio, text, meta)

    assert fused.ndim == 1
    assert fused.shape[0] == 128
    assert np.isfinite(fused).all()
