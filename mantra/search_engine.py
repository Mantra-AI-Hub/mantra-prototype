"""Semantic audio search over persisted embeddings with filtering and cache."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from mantra.ann_router import ANNRouter
from mantra.coarse_index import CoarseIndex
import numpy as np
from mantra.audio_engine.audio_loader import load_audio
from mantra.audio_engine.feature_extractor import (
    extract_chroma,
    extract_pitch_contour,
    extract_spectral_features,
    extract_tempo,
)
from mantra.cache import LRUCache
from mantra.database.track_store import TrackStore
from mantra.reranker import rerank_results
from mantra.vector_quantizer import VectorQuantizer
from mantra.vector_engine.embedding_builder import EMBEDDING_SIZE, build_music_embedding
from mantra.vector_index import VectorIndex


SearchFilters = Dict[str, object]

_SEARCH_CACHE: LRUCache[str, List[Dict[str, object]]] = LRUCache(capacity=512)
_ANN_STATE: Dict[str, object] = {}


def _build_query_embedding(query_audio_path: str):
    audio = load_audio(query_audio_path)
    features = {
        "chroma": extract_chroma(audio),
        "tempo": extract_tempo(audio),
        "pitch_contour": extract_pitch_contour(audio),
        "spectral": extract_spectral_features(audio),
    }
    return build_music_embedding(features)


def _normalize_text(value: object) -> str:
    return str(value or "").strip().lower()


def _metadata_matches_filters(metadata: Dict[str, object], filters: Optional[SearchFilters]) -> bool:
    if not filters:
        return True

    for key, expected in filters.items():
        if key not in {"artist", "album", "genre", "duration", "tags", "year"}:
            continue

        actual = metadata.get(key)
        if key == "tags":
            expected_tags: List[str]
            if isinstance(expected, list):
                expected_tags = [_normalize_text(v) for v in expected]
            else:
                expected_tags = [_normalize_text(v) for v in str(expected).split(",") if v.strip()]
            actual_tags = {_normalize_text(v) for v in (actual or [])}
            if not all(tag in actual_tags for tag in expected_tags):
                return False
            continue

        if key == "year":
            try:
                if int(actual) != int(expected):
                    return False
            except (TypeError, ValueError):
                return False
            continue

        if key == "duration":
            try:
                if float(actual) < float(expected):
                    return False
            except (TypeError, ValueError):
                return False
            continue

        if _normalize_text(actual) != _normalize_text(expected):
            return False

    return True


def _cache_key(query_audio_path: str, top_k: int, filters: Optional[SearchFilters]) -> str:
    query_bytes = Path(query_audio_path).read_bytes()
    audio_hash = hashlib.sha1(query_bytes).hexdigest()
    filter_repr = repr(sorted((filters or {}).items()))
    return f"{audio_hash}:{int(top_k)}:{filter_repr}"


def _post_filter_results(
    raw_matches: Iterable[tuple[str, float]],
    track_store: TrackStore,
    top_k: int,
    filters: Optional[SearchFilters],
) -> List[Dict[str, object]]:
    filtered: List[Dict[str, object]] = []
    for track_id, score in raw_matches:
        metadata = track_store.get_track(track_id) or {}
        if not _metadata_matches_filters(metadata, filters):
            continue
        filtered.append({"track_id": track_id, "score": score, "metadata": metadata})
        if len(filtered) >= top_k:
            break
    return filtered


def _safe_cosine(a, b) -> float:
    va = np.asarray(a, dtype=np.float32).reshape(-1)
    vb = np.asarray(b, dtype=np.float32).reshape(-1)
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
    if denom == 0:
        return 0.0
    return float(np.dot(va, vb) / denom)


def _init_ann_state(index) -> bool:
    base = getattr(index, "_index", None)
    vectors = np.asarray(getattr(base, "_vectors", np.empty((0, EMBEDDING_SIZE), dtype=np.float32)), dtype=np.float32)
    track_ids = list(getattr(base, "_track_ids", []))
    if vectors.ndim != 2 or vectors.shape[0] == 0 or len(track_ids) != vectors.shape[0]:
        return False

    router = ANNRouter(clusters=min(16, max(1, vectors.shape[0])))
    coarse = CoarseIndex(clusters=min(32, max(1, vectors.shape[0])))
    quantizer = VectorQuantizer(subspaces=min(4, max(1, vectors.shape[1])), codebook_size=16)

    router.train(vectors)
    coarse.train(vectors)
    quantizer.train(vectors)

    _ANN_STATE["vectors"] = vectors
    _ANN_STATE["track_ids"] = track_ids
    _ANN_STATE["router"] = router
    _ANN_STATE["coarse"] = coarse
    _ANN_STATE["quantizer"] = quantizer
    return True


def _two_stage_ann_search(query_embedding, top_k: int):
    vectors = _ANN_STATE["vectors"]
    track_ids = _ANN_STATE["track_ids"]
    router = _ANN_STATE["router"]
    coarse = _ANN_STATE["coarse"]
    quantizer = _ANN_STATE["quantizer"]

    route_clusters = set(router.route(query_embedding, top_shards=3))
    candidate_clusters = set(coarse.get_candidate_clusters(query_embedding, n=5))
    selected = route_clusters | candidate_clusters
    if not selected:
        selected = set(range(coarse.centroids.shape[0]))

    candidate_indices = [
        idx for idx, cid in enumerate(coarse.assignments.tolist()) if int(cid) in selected
    ]
    if not candidate_indices:
        candidate_indices = list(range(vectors.shape[0]))

    scored = []
    for idx in candidate_indices:
        v = vectors[idx]
        code = quantizer.encode(v)
        approx = quantizer.decode(code)
        score = _safe_cosine(query_embedding, approx)
        scored.append((track_ids[idx], float(score)))
    scored.sort(key=lambda item: item[1], reverse=True)
    return scored[: max(int(top_k), int(top_k) * 10)]


def semantic_search(
    query_audio_path: str,
    top_k: int,
    vector_index_path: str,
    track_store: TrackStore,
    filters: Optional[SearchFilters] = None,
) -> List[Dict[str, object]]:
    """Search top-k nearest tracks for a query audio file path."""
    path = Path(query_audio_path)
    if not path.exists():
        raise FileNotFoundError(f"Query audio file not found: {query_audio_path}")
    if path.suffix.lower() != ".wav":
        raise ValueError("Only WAV files are supported for semantic search")

    key = _cache_key(query_audio_path, top_k, filters)
    cached = _SEARCH_CACHE.get(key)
    if cached is not None:
        return cached

    query_embedding = _build_query_embedding(query_audio_path)

    try:
        index = VectorIndex.load(vector_index_path)
    except FileNotFoundError:
        index = VectorIndex(dimension=EMBEDDING_SIZE)

    if _init_ann_state(index):
        matches = _two_stage_ann_search(query_embedding, top_k=top_k)
    else:
        candidate_k = max(int(top_k), int(top_k) * 10)
        matches = index.search(query_embedding, k=candidate_k)
    results = _post_filter_results(matches, track_store=track_store, top_k=top_k, filters=filters)
    results = rerank_results(results, filters=filters)[: max(0, int(top_k))]
    _SEARCH_CACHE.set(key, results)
    return results


def semantic_search_batch(
    queries: List[Dict[str, object]],
    top_k: int,
    vector_index_path: str,
    track_store: TrackStore,
) -> List[Dict[str, object]]:
    """Process multiple semantic searches in one call."""
    try:
        index = VectorIndex.load(vector_index_path)
    except FileNotFoundError:
        index = VectorIndex(dimension=EMBEDDING_SIZE)

    response: List[Dict[str, object]] = []
    for query in queries:
        query_audio_path = str(query["query_audio_path"])
        filters = query.get("filters")
        path = Path(query_audio_path)
        if not path.exists():
            response.append({"query_audio_path": query_audio_path, "results": []})
            continue
        if path.suffix.lower() != ".wav":
            response.append({"query_audio_path": query_audio_path, "results": []})
            continue

        key = _cache_key(query_audio_path, top_k, filters if isinstance(filters, dict) else None)
        cached = _SEARCH_CACHE.get(key)
        if cached is not None:
            response.append({"query_audio_path": query_audio_path, "results": cached})
            continue

        query_embedding = _build_query_embedding(query_audio_path)
        matches = index.search(query_embedding, k=max(int(top_k), int(top_k) * 10))
        query_results = _post_filter_results(
            matches,
            track_store=track_store,
            top_k=top_k,
            filters=filters if isinstance(filters, dict) else None,
        )
        query_results = rerank_results(
            query_results,
            filters=filters if isinstance(filters, dict) else None,
        )[: max(0, int(top_k))]
        _SEARCH_CACHE.set(key, query_results)
        response.append({"query_audio_path": query_audio_path, "results": query_results})

    return response


def search_cache_metrics() -> Dict[str, int]:
    return _SEARCH_CACHE.metrics()


def clear_search_cache() -> None:
    _SEARCH_CACHE.clear()
