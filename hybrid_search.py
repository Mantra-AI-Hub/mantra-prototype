"""Hybrid search combining fingerprint recognition and embedding similarity."""

from __future__ import annotations

from typing import Dict, List

from mantra.fingerprint_engine import generate_fingerprints
from mantra.fingerprint_index import FingerprintIndex
from mantra.search_engine import semantic_search


def hybrid_search(
    query_audio_path: str,
    top_k: int,
    fingerprint_index: FingerprintIndex,
    vector_index_path: str,
    track_store,
) -> List[Dict[str, object]]:
    fp = generate_fingerprints(query_audio_path)
    fp_results = fingerprint_index.query(fp)
    sem_results = semantic_search(
        query_audio_path=query_audio_path,
        top_k=max(1, int(top_k) * 2),
        vector_index_path=vector_index_path,
        track_store=track_store,
    )

    by_track: Dict[str, Dict[str, object]] = {}
    for item in sem_results:
        track_id = str(item["track_id"])
        by_track[track_id] = {
            "track_id": track_id,
            "score": float(item.get("score") or item.get("final_score") or 0.0) * 0.6,
            "metadata": item.get("metadata") or {},
        }

    for item in fp_results:
        track_id = str(item["track_id"])
        base = by_track.get(track_id, {"track_id": track_id, "score": 0.0, "metadata": track_store.get_track(track_id) or {}})
        base["score"] = float(base["score"]) + float(item.get("score", 0.0)) * 0.4
        base["offset"] = float(item.get("offset", 0.0))
        by_track[track_id] = base

    merged = list(by_track.values())
    merged.sort(key=lambda value: value.get("score", 0.0), reverse=True)
    return merged[: max(0, int(top_k))]
