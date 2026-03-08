"""Text and lyrics embedding utilities."""

from __future__ import annotations

import hashlib
from typing import List

import numpy as np

_EMBED_DIM = 128

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception:  # pragma: no cover
    SentenceTransformer = None

_MODEL = None


def _get_model():
    global _MODEL
    if SentenceTransformer is None:
        return None
    if _MODEL is None:
        try:
            _MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        except Exception:
            _MODEL = None
    return _MODEL


def _hash_embedding(text: str) -> np.ndarray:
    text = str(text or "").strip().lower()
    if not text:
        return np.zeros(_EMBED_DIM, dtype=np.float32)
    vec = np.zeros(_EMBED_DIM, dtype=np.float32)
    for token in text.split():
        digest = hashlib.sha1(token.encode("utf-8")).digest()
        for i, byte in enumerate(digest):
            vec[(i + byte) % _EMBED_DIM] += float(byte) / 255.0
    norm = float(np.linalg.norm(vec))
    if norm > 0:
        vec /= norm
    return vec


def compute_text_embedding(text: str) -> np.ndarray:
    model = _get_model()
    if model is not None:
        arr = np.asarray(model.encode([text])[0], dtype=np.float32)
        norm = float(np.linalg.norm(arr))
        if norm > 0:
            arr /= norm
        return arr
    return _hash_embedding(text)


def compute_lyrics_embedding(lyrics: str) -> np.ndarray:
    return compute_text_embedding(lyrics)
