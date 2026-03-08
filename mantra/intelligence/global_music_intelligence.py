"""Global music intelligence backed by synthetic listeners and evolution experiments."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np

from mantra.intelligence.evolution_engine import EvolutionEngine, TrackFitness, TrackGenome

if TYPE_CHECKING:
    from mantra.feature_store import FeatureStore


@dataclass
class MusicFeatureVector:
    tempo: float
    energy: float
    mood: float
    novelty: float
    rhythm_complexity: float
    harmonic_density: float
    emotional_intensity: float


@dataclass
class EngagementSnapshot:
    track_id: str
    completion_rate: float
    replay_rate: float
    skip_rate: float
    virality_score: float
    listener_satisfaction: float


@dataclass
class TrendPattern:
    feature_name: str
    correlation_score: float
    impact_strength: float


@dataclass
class HitPrediction:
    track_id: str
    hit_probability: float
    expected_virality: float


class GlobalMusicIntelligenceEngine:
    def __init__(self, seed: int = 0, feature_store: Optional["FeatureStore"] = None):
        self._rng = np.random.default_rng(seed)
        self.feature_store = feature_store or self._create_feature_store()
        self.feature_vectors: List[MusicFeatureVector] = []
        self.engagement_snapshots: List[EngagementSnapshot] = []
        self.feature_correlations: Dict[str, Dict[str, float]] = {}
        self.trend_patterns: List[TrendPattern] = []
        self._data_ingested = False

    def _create_feature_store(self) -> Optional["FeatureStore"]:
        try:
            from mantra.feature_store import FeatureStore as Store

            return Store()
        except Exception:
            return None

    def ingest_experiment_data(
        self,
        experiment_results: Optional[List[Tuple[TrackGenome, TrackFitness]]] = None,
        population_size: int = 16,
        generations: int = 2,
    ) -> None:
        if self._data_ingested and experiment_results is None:
            return
        dataset: List[Tuple[TrackGenome, TrackFitness]] = experiment_results or self._run_evolution_simulation(
            population_size=population_size, generations=generations
        )
        if not dataset:
            return
        self._populate_from_dataset(dataset)

    def _run_evolution_simulation(self, population_size: int, generations: int) -> List[Tuple[TrackGenome, TrackFitness]]:
        engine = EvolutionEngine(seed=int(self._rng.integers(0, 1 << 31)))
        population = engine.initialize_population(population_size)
        dataset: List[Tuple[TrackGenome, TrackFitness]] = []
        for _ in range(max(1, generations)):
            previous_population = list(population)
            population, fitnesses = engine.run_generation()
            dataset.extend(list(zip(previous_population, fitnesses)))
        return dataset

    def _populate_from_dataset(self, dataset: List[Tuple[TrackGenome, TrackFitness]]) -> None:
        self.feature_vectors = []
        self.engagement_snapshots = []
        for genome, fitness in dataset:
            vector = self._vector_from_genome(genome)
            snapshot = EngagementSnapshot(
                track_id=fitness.track_id or genome.track_id,
                completion_rate=fitness.completion_rate,
                replay_rate=fitness.replay_rate,
                skip_rate=fitness.skip_rate,
                virality_score=fitness.virality,
                listener_satisfaction=float(
                    np.clip(fitness.virality + fitness.completion_rate - fitness.skip_rate, 0.0, 1.0)
                ),
            )
            self.feature_vectors.append(vector)
            self.engagement_snapshots.append(snapshot)
        self._data_ingested = True

    def _vector_from_genome(self, genome: TrackGenome) -> MusicFeatureVector:
        emotional = float(
            np.clip((genome.energy + genome.mood + genome.harmonic_density) / 3.0, 0.0, 1.0)
        )
        return MusicFeatureVector(
            tempo=genome.tempo,
            energy=genome.energy,
            mood=genome.mood,
            novelty=genome.novelty,
            rhythm_complexity=genome.rhythm_complexity,
            harmonic_density=genome.harmonic_density,
            emotional_intensity=emotional,
        )

    def extract_feature_vectors(self) -> List[MusicFeatureVector]:
        if not self.feature_vectors:
            self.ingest_experiment_data()
        return list(self.feature_vectors)

    def compute_feature_correlations(self) -> Dict[str, Dict[str, float]]:
        if not self.feature_vectors or not self.engagement_snapshots:
            self.feature_correlations = {}
            return self.feature_correlations
        feature_names = [
            "tempo",
            "energy",
            "mood",
            "novelty",
            "rhythm_complexity",
            "harmonic_density",
            "emotional_intensity",
        ]
        metric_names = ["virality_score", "completion_rate", "replay_rate", "skip_rate", "listener_satisfaction"]
        feature_matrix = np.array(
            [[getattr(vec, name) for name in feature_names] for vec in self.feature_vectors], dtype=float
        )
        metric_matrix = np.array(
            [[getattr(snap, name) for name in metric_names] for snap in self.engagement_snapshots], dtype=float
        )
        correlations: Dict[str, Dict[str, float]] = {}
        for idx, feature_name in enumerate(feature_names):
            feature_col = feature_matrix[:, idx]
            correlations[feature_name] = {}
            for metric_idx, metric_name in enumerate(metric_names):
                metric_col = metric_matrix[:, metric_idx]
                correlations[feature_name][metric_name] = self._safe_correlation(feature_col, metric_col)
        self.feature_correlations = correlations
        return correlations

    def _safe_correlation(self, feature_col: np.ndarray, metric_col: np.ndarray) -> float:
        if feature_col.size < 2 or metric_col.size < 2:
            return 0.0
        if np.allclose(feature_col, feature_col[0]) or np.allclose(metric_col, metric_col[0]):
            return 0.0
        try:
            corr = np.corrcoef(feature_col, metric_col)[0, 1]
        except FloatingPointError:
            return 0.0
        if np.isnan(corr):
            return 0.0
        return float(np.clip(corr, -1.0, 1.0))

    def identify_trending_patterns(self, top_k: int = 3) -> Dict[str, List[TrendPattern]]:
        correlations = self.compute_feature_correlations()
        if not correlations:
            self.trend_patterns = []
            return {"positive": [], "negative": []}
        positive: List[TrendPattern] = []
        negative: List[TrendPattern] = []
        for feature_name, metric_values in correlations.items():
            avg_corr = float(sum(metric_values.values()) / max(1, len(metric_values)))
            impact = float(np.clip(sum(abs(v) for v in metric_values.values()), 0.0, 1.0))
            pattern = TrendPattern(feature_name=feature_name, correlation_score=avg_corr, impact_strength=impact)
            if avg_corr >= 0:
                positive.append(pattern)
            else:
                negative.append(pattern)
        positive.sort(key=lambda pattern: pattern.correlation_score, reverse=True)
        negative.sort(key=lambda pattern: pattern.correlation_score)
        positive_slice = positive[:top_k]
        negative_slice = negative[:top_k]
        self.trend_patterns = positive_slice + negative_slice
        self._persist_patterns(positive_slice, negative_slice)
        return {"positive": positive_slice, "negative": negative_slice}

    def _persist_patterns(self, positive: List[TrendPattern], negative: List[TrendPattern]) -> None:
        if not self.feature_store or not (positive or negative):
            return
        payload = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "positive": [pattern.__dict__ for pattern in positive],
            "negative": [pattern.__dict__ for pattern in negative],
        }
        try:
            self.feature_store.put("global_music_trends", payload)
        except Exception:
            pass

    def predict_hit_probability(self, feature_vector: MusicFeatureVector) -> float:
        correlations = self.feature_correlations or self.compute_feature_correlations()
        if not correlations:
            return 0.5
        score = 0.0
        weight_sum = 0.0
        for feature_name, metric_values in correlations.items():
            feature_value = getattr(feature_vector, feature_name, 0.0)
            weight = (
                metric_values.get("virality_score", 0.0) * 0.6
                + metric_values.get("completion_rate", 0.0) * 0.2
                + metric_values.get("replay_rate", 0.0) * 0.2
                - metric_values.get("skip_rate", 0.0) * 0.3
                + metric_values.get("listener_satisfaction", 0.0) * 0.1
            )
            score += feature_value * weight
            weight_sum += abs(weight)
        normalized = score / max(weight_sum, 1e-6)
        baseline = 0.5 + normalized * 0.25
        return float(np.clip(baseline, 0.0, 1.0))

    def estimate_expected_virality(self, feature_vector: MusicFeatureVector) -> float:
        probability = self.predict_hit_probability(feature_vector)
        avg_virality = (
            float(np.mean([snapshot.virality_score for snapshot in self.engagement_snapshots]))
            if self.engagement_snapshots
            else 0.0
        )
        expected = probability * 0.6 + avg_virality * 0.4
        return float(np.clip(expected, 0.0, 1.0))

    def generate_trend_report(self, top_k: int = 3) -> Dict[str, object]:
        self.ingest_experiment_data()
        patterns = self.identify_trending_patterns(top_k=top_k)
        vectors = [vec.__dict__ for vec in self.feature_vectors[: max(0, min(len(self.feature_vectors), top_k))]]
        engagements = [snap.__dict__ for snap in self.engagement_snapshots[: max(0, min(len(self.engagement_snapshots), top_k))]]
        return {
            "top_positive_features": [pattern.__dict__ for pattern in patterns["positive"]],
            "top_negative_features": [pattern.__dict__ for pattern in patterns["negative"]],
            "current_trend_vectors": vectors,
            "sampled_engagement": engagements,
            "population": len(self.feature_vectors),
        }

    def describe_snapshot(self, snapshot: EngagementSnapshot) -> Dict[str, object]:
        return snapshot.__dict__

    def map_engagement(self) -> Dict[str, object]:
        return {
            snapshot.track_id: {
                "completion_rate": snapshot.completion_rate,
                "replay_rate": snapshot.replay_rate,
                "skip_rate": snapshot.skip_rate,
                "virality": snapshot.virality_score,
            }
            for snapshot in self.engagement_snapshots
        }
