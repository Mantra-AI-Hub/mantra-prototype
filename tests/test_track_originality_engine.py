from mantra.intelligence.global_music_intelligence import GlobalMusicIntelligenceEngine
from mantra.intelligence.track_originality_engine import TrackOriginalityEngine


def test_extract_fingerprint_prefers_explicit_vectors():
    engine = TrackOriginalityEngine()
    fingerprint = engine.extract_fingerprint(
        {
            "track_id": "track_a",
            "melody_vector": [1.0, 0.0, 1.0],
            "harmony_vector": [0.5, 0.25],
            "rhythm_vector": [0.2, 0.4, 0.6],
            "energy_vector": [0.9, 0.1],
        }
    )
    assert fingerprint.track_id == "track_a"
    assert fingerprint.melody_vector == [1.0, 0.0, 1.0]
    assert fingerprint.harmony_vector == [0.5, 0.25]


def test_similarity_calculation_is_weighted_and_bounded():
    engine = TrackOriginalityEngine()
    base = engine.extract_fingerprint(
        {
            "track_id": "base",
            "melody_vector": [1.0, 0.0, 1.0],
            "harmony_vector": [1.0, 0.0],
            "rhythm_vector": [0.5, 0.5],
            "energy_vector": [0.25, 0.75],
        }
    )
    duplicate = engine.extract_fingerprint(
        {
            "track_id": "dup",
            "melody_vector": [1.0, 0.0, 1.0],
            "harmony_vector": [1.0, 0.0],
            "rhythm_vector": [0.5, 0.5],
            "energy_vector": [0.25, 0.75],
        }
    )
    similarity = engine.compute_similarity(base, duplicate)
    assert similarity == 1.0


def test_originality_bounds_with_feature_vector_fallback():
    engine = TrackOriginalityEngine()
    candidate = engine.extract_fingerprint({"track_id": "candidate", "feature_vector": [0.1, 0.2, 0.3, 0.4]})
    library = [
        engine.extract_fingerprint({"track_id": "lib_a", "feature_vector": [0.4, 0.3, 0.2, 0.1]}),
        engine.extract_fingerprint({"track_id": "lib_b", "feature_vector": [0.1, 0.2, 0.35, 0.45]}),
    ]
    result = engine.compute_originality(candidate, library)
    assert 0.0 <= result.similarity_score <= 1.0
    assert 0.0 <= result.originality_score <= 1.0
    assert result.most_similar_track in {"lib_a", "lib_b"}


def test_duplicate_detection_marks_track_as_risky():
    engine = TrackOriginalityEngine()
    fingerprint = engine.extract_fingerprint({"track_id": "track_dup", "feature_vector": [0.2, 0.4, 0.6, 0.8]})
    result = engine.compute_originality(
        fingerprint,
        [engine.extract_fingerprint({"track_id": "track_ref", "feature_vector": [0.2, 0.4, 0.6, 0.8]})],
    )
    assert result.similarity_score >= 0.8
    assert result.originality_score <= 0.2
    assert result.most_similar_track == "track_ref"


def test_global_music_intelligence_originality_assessment():
    engine = GlobalMusicIntelligenceEngine(seed=7)
    engine.ingest_experiment_data(population_size=6, generations=1)
    result = engine.assess_originality({"track_id": "track_candidate"})
    assert 0.0 <= result.similarity_score <= 1.0
    assert 0.0 <= result.originality_score <= 1.0
