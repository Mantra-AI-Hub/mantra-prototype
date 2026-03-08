"""Playlist generation utilities."""

from __future__ import annotations

from typing import Dict, List


def generate_playlist(seed_track_id: str, length: int, recommendation_engine, track_store) -> List[Dict[str, object]]:
    length = max(1, int(length))
    items: List[Dict[str, object]] = []

    seed_meta = track_store.get_track(seed_track_id)
    if seed_meta:
        items.append({"track_id": seed_track_id, "metadata": seed_meta})

    recs = recommendation_engine.recommend(seed_track_id, k=max(0, length - len(items)))
    for track_id, score in recs:
        if len(items) >= length:
            break
        items.append({"track_id": track_id, "score": float(score), "metadata": track_store.get_track(track_id)})

    return items[:length]
