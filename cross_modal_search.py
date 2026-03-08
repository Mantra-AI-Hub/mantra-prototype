"""Cross-modal search from text or lyrics queries."""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from mantra.text_embedding_engine import compute_lyrics_embedding, compute_text_embedding
from mantra.vector_index import VectorIndex


def _text_to_audio_dim(text_vec, target_dim: int) -> np.ndarray:
    vec = np.asarray(text_vec, dtype=np.float32).reshape(-1)
    out = np.zeros(target_dim, dtype=np.float32)
    if vec.size >= target_dim:
        out[:] = vec[:target_dim]
    else:
        out[: vec.size] = vec
    norm = float(np.linalg.norm(out))
    if norm > 0:
        out /= norm
    return out


def _metadata_bonus(query_text: str, metadata: Dict[str, object]) -> float:
    q = str(query_text or "").lower()
    if not q:
        return 0.0
    score = 0.0
    for key in ("artist", "album", "genre"):
        value = str(metadata.get(key) or "").lower()
        if value and value in q:
            score += 0.1
    tags = metadata.get("tags") or []
    for tag in tags:
        if str(tag).lower() in q:
            score += 0.05
    return min(0.25, score)


def _search(query_text: str, top_k: int, vector_index_path: str, track_store, mode: str) -> List[Dict[str, object]]:
    text_vec = compute_lyrics_embedding(query_text) if mode == "lyrics" else compute_text_embedding(query_text)

    tracks = track_store.list_tracks()
    if tracks:
        target_dim = int(len(np.zeros(87, dtype=np.float32)))
        emb_path = str(tracks[0].get("embedding_path") or vector_index_path)
    else:
        target_dim = 87
        emb_path = vector_index_path

    query_vec = _text_to_audio_dim(text_vec, target_dim)
    try:
        index = VectorIndex.load(emb_path)
    except FileNotFoundError:
        index = VectorIndex(dimension=target_dim)

    matches = index.search(query_vec, k=max(int(top_k), int(top_k) * 3))
    results = []
    for track_id, score in matches:
        metadata = track_store.get_track(track_id) or {}
        final = float(score) + _metadata_bonus(query_text, metadata)
        results.append({"track_id": track_id, "score": final, "metadata": metadata})

    results.sort(key=lambda item: item["score"], reverse=True)
    return results[: max(0, int(top_k))]


def search_by_text(query_text: str, top_k: int, vector_index_path: str, track_store) -> List[Dict[str, object]]:
    return _search(query_text, top_k, vector_index_path, track_store, mode="text")


def search_by_lyrics(lyrics: str, top_k: int, vector_index_path: str, track_store) -> List[Dict[str, object]]:
    return _search(lyrics, top_k, vector_index_path, track_store, mode="lyrics")
