import numpy as np

from mantra.text_embedding_engine import compute_lyrics_embedding, compute_text_embedding


def test_text_embedding_deterministic_and_finite():
    a = compute_text_embedding("ambient chill piano")
    b = compute_text_embedding("ambient chill piano")
    l = compute_lyrics_embedding("hello darkness my old friend")

    assert a.shape == b.shape
    assert np.allclose(a, b)
    assert np.isfinite(a).all()
    assert np.isfinite(l).all()
