"""Real-time candidate generation by combining multiple recommenders."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

from mantra.intelligence.music_foundation_model import MusicFoundationModel
from mantra.intelligence.music_genome_store import MusicGenomeStore


class RealtimeRecommender:
    def __init__(
        self,
        vector_index_service,
        graph_recommender,
        transformer_session_model,
        trend_detector,
        genome_store: MusicGenomeStore | None = None,
        foundation_model: MusicFoundationModel | None = None,
    ):
        self.vector_index_service = vector_index_service
        self.graph_recommender = graph_recommender
        self.transformer_session_model = transformer_session_model
        self.trend_detector = trend_detector
        self.genome_store = genome_store
        self.foundation_model = foundation_model or MusicFoundationModel()

    def generate_candidates(
        self,
        user_id: str,
        user_embedding: Sequence[float] | None,
        session_history: List[str],
        top_k: int = 30,
    ) -> Dict[str, float]:
        limit = max(1, int(top_k))
        scores: Dict[str, float] = {}

        if user_embedding:
            try:
                vector_results = self.vector_index_service.search(user_embedding, limit)
            except Exception:
                vector_results = []
            for track_id, score in vector_results:
                scores[str(track_id)] = max(scores.get(str(track_id), 0.0), float(score))

        for track_id, score in self.graph_recommender.recommend_from_graph(user_id, limit):
            scores[str(track_id)] = max(scores.get(str(track_id), 0.0), float(score))

        for track_id, score in self.transformer_session_model.predict_next_tracks(user_id, session_history, top_k=limit):
            scores[str(track_id)] = max(scores.get(str(track_id), 0.0), float(score))

        for idx, item in enumerate(self.trend_detector.detect_trending_tracks(top_k=limit)):
            track_id = str(item.get("track_id") or "")
            if not track_id:
                continue
            trend_score = float(item.get("count", 0)) / float(idx + 1)
            scores[track_id] = max(scores.get(track_id, 0.0), trend_score)

        if self.genome_store is not None and session_history:
            seed_track = str(session_history[-1])
            seed_genome = self.genome_store.get_genome(seed_track)
            if seed_genome:
                for item in self.genome_store.search_similar(seed_genome, top_k=limit):
                    track_id = str(item.get("track_id") or "")
                    if not track_id:
                        continue
                    score = 0.2 * float(item.get("genome_similarity", 0.0)) + 0.05 * float(item.get("score", 0.0))
                    scores[track_id] = max(scores.get(track_id, 0.0), score)

        return scores

    def top_candidates(
        self,
        user_id: str,
        user_embedding: Sequence[float] | None,
        session_history: List[str],
        top_k: int = 10,
    ) -> List[Tuple[str, float]]:
        scores = self.generate_candidates(user_id, user_embedding, session_history, top_k=max(10, top_k * 3))
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [(track_id, float(score)) for track_id, score in ranked[: max(0, int(top_k))]]

