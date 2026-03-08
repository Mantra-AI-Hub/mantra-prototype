import numpy as np

from mantra.vector_engine.embedding_builder import EMBEDDING_SIZE, build_music_embedding


def test_build_music_embedding_fixed_size():
    features = {
        "chroma": np.ones(12, dtype=np.float32),
        "tempo": 120.0,
        "pitch_contour": np.linspace(0.0, 1.0, 64, dtype=np.float32),
        "spectral": {
            "centroid_mean": 1000.0,
            "centroid_std": 50.0,
            "bandwidth_mean": 800.0,
            "bandwidth_std": 40.0,
            "rolloff_mean": 3000.0,
            "rolloff_std": 100.0,
            "zcr_mean": 0.1,
            "zcr_std": 0.01,
            "rms_mean": 0.2,
            "rms_std": 0.03,
        },
    }

    embedding = build_music_embedding(features)

    assert embedding.shape == (EMBEDDING_SIZE,)
    assert embedding.dtype == np.float32


def test_build_music_embedding_handles_missing_features():
    embedding = build_music_embedding({})

    assert embedding.shape == (EMBEDDING_SIZE,)
    assert np.all(np.isfinite(embedding))
    assert np.allclose(embedding, 0.0)
