"""Fusion utilities for multimodal music embeddings."""

from __future__ import annotations

import numpy as np


def _to_vec(vec) -> np.ndarray:
    return np.asarray(vec, dtype=np.float32).reshape(-1)


def fuse_embeddings(audio_vec, text_vec, metadata_vec) -> np.ndarray:
    a = _to_vec(audio_vec)
    t = _to_vec(text_vec)
    m = _to_vec(metadata_vec)

    dim = max(a.size, t.size, m.size)
    out_a = np.zeros(dim, dtype=np.float32)
    out_t = np.zeros(dim, dtype=np.float32)
    out_m = np.zeros(dim, dtype=np.float32)

    out_a[: a.size] = a
    out_t[: t.size] = t
    out_m[: m.size] = m

    fused = 0.6 * out_a + 0.3 * out_t + 0.1 * out_m
    norm = float(np.linalg.norm(fused))
    if norm > 0:
        fused /= norm
    return fused.astype(np.float32)
