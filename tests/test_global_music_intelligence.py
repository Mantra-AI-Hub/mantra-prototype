"""Tests for the global music intelligence engine."""

from mantra.intelligence.global_music_intelligence import GlobalMusicIntelligenceEngine, MusicFeatureVector


def test_feature_vectors_and_correlations():
    engine = GlobalMusicIntelligenceEngine(seed=1)
    engine.ingest_experiment_data(population_size=8, generations=1)
    vectors = engine.extract_feature_vectors()
    assert vectors
    assert all(0.0 <= vec.tempo <= 180.0 for vec in vectors)
    correlations = engine.compute_feature_correlations()
    assert "tempo" in correlations
    assert isinstance(correlations["tempo"]["virality_score"], float)


def test_hit_probability_bounds():
    engine = GlobalMusicIntelligenceEngine(seed=2)
    engine.ingest_experiment_data(population_size=6, generations=1)
    vector = MusicFeatureVector(
        tempo=120.0,
        energy=0.6,
        mood=0.7,
        novelty=0.45,
        rhythm_complexity=0.5,
        harmonic_density=0.55,
        emotional_intensity=0.65,
    )
    probability = engine.predict_hit_probability(vector)
    assert 0.0 <= probability <= 1.0


def test_trend_report_structure():
    engine = GlobalMusicIntelligenceEngine(seed=3)
    engine.ingest_experiment_data(population_size=6, generations=1)
    report = engine.generate_trend_report(top_k=2)
    assert "top_positive_features" in report
    assert isinstance(report["top_positive_features"], list)
    assert "top_negative_features" in report
    assert isinstance(report["top_negative_features"], list)
    assert "current_trend_vectors" in report
    assert isinstance(report["current_trend_vectors"], list)
