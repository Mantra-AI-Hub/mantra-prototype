"""Discovery feed ranking that fuses multiple recommendation signals."""

from __future__ import annotations

from typing import Dict, Iterable, List


class DiscoveryFeedRanker:
    def rank(
        self,
        user_id: str,
        candidates: Iterable[Dict[str, object]],
        cluster_pref_tracks: set[str] | None = None,
    ) -> List[Dict[str, object]]:
        cluster_pref_tracks = cluster_pref_tracks or set()
        ranked = []
        for item in candidates:
            row = dict(item)
            base = float(row.get("score", 0.0))
            rank_score = float(row.get("rank_score", 0.0))
            rl_score = float(row.get("rl_score", 0.0))
            bandit_score = float(row.get("bandit_score", 0.0))
            trend = float(row.get("trending_score", 0.0))
            engagement = float(row.get("engagement_score", 0.0))
            genome_similarity = float(row.get("genome_similarity", 0.0))
            foundation_similarity = float(row.get("foundation_embedding_similarity", 0.0))
            track_id = str(row.get("track_id"))
            cluster_boost = 0.1 if track_id in cluster_pref_tracks else 0.0
            final_score = (
                0.28 * rank_score
                + 0.18 * base
                + 0.12 * rl_score
                + 0.12 * bandit_score
                + 0.08 * trend
                + 0.05 * engagement
                + 0.12 * genome_similarity
                + 0.05 * foundation_similarity
                + cluster_boost
            )
            row["discovery_score"] = float(final_score)
            ranked.append(row)
        ranked.sort(key=lambda x: x.get("discovery_score", 0.0), reverse=True)
        return ranked

