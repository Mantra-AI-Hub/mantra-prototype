"""Search result re-ranking using similarity + metadata + popularity."""

from __future__ import annotations

from typing import Dict, List, Optional


def _metadata_match_score(metadata: Dict[str, object], filters: Optional[Dict[str, object]]) -> float:
    if not filters:
        return 1.0
    matched = 0
    total = 0
    for key, expected in filters.items():
        if key not in metadata:
            continue
        total += 1
        actual = metadata.get(key)
        if str(actual).lower() == str(expected).lower():
            matched += 1
    if total == 0:
        return 0.0
    return float(matched / total)


def _popularity_score(metadata: Dict[str, object]) -> float:
    # Heuristic: more fingerprint hashes implies richer content/popularity proxy.
    count = int(metadata.get("fingerprint_hash_count") or 0)
    return min(1.0, count / 500.0)


def rerank_results(results: List[Dict[str, object]], filters: Optional[Dict[str, object]] = None) -> List[Dict[str, object]]:
    reranked: List[Dict[str, object]] = []
    for item in results:
        similarity = float(item.get("score", 0.0))
        metadata = item.get("metadata") or {}
        mscore = _metadata_match_score(metadata, filters)
        pscore = _popularity_score(metadata)
        final_score = similarity * 0.8 + mscore * 0.1 + pscore * 0.1
        enriched = dict(item)
        enriched["final_score"] = float(final_score)
        reranked.append(enriched)

    reranked.sort(key=lambda value: value.get("final_score", 0.0), reverse=True)
    return reranked
