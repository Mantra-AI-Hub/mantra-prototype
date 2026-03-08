"""Ranking utilities for recommendation candidates."""

from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np


def _cosine(a, b) -> float:
    va = np.asarray(a, dtype=np.float32).reshape(-1)
    vb = np.asarray(b, dtype=np.float32).reshape(-1)
    if va.size != vb.size:
        size = max(va.size, vb.size)
        pa = np.zeros(size, dtype=np.float32)
        pb = np.zeros(size, dtype=np.float32)
        pa[: va.size] = va
        pb[: vb.size] = vb
        va, vb = pa, pb
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
    if denom == 0:
        return 0.0
    return float(np.dot(va, vb) / denom)


def rank_tracks(user_features: Dict[str, object], candidate_tracks: Iterable[Dict[str, object]]) -> List[Dict[str, object]]:
    user_vec = np.asarray(user_features.get("embedding") or [], dtype=np.float32)
    ranked: List[Dict[str, object]] = []

    for track in candidate_tracks:
        track_vec = np.asarray(track.get("embedding") or [], dtype=np.float32)
        score = _cosine(user_vec, track_vec)
        item = dict(track)
        item["rank_score"] = float(score)
        ranked.append(item)

    ranked.sort(key=lambda x: x.get("rank_score", 0.0), reverse=True)
    return ranked
