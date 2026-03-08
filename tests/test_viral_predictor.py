from mantra.viral_predictor import ViralPredictor


def test_viral_predictor_returns_probability_like_score():
    predictor = ViralPredictor()

    low = predictor.predict_track_virality({"play_count": 10, "velocity": 0.1, "like_rate": 0.1})
    high = predictor.predict_track_virality({"play_count": 10000, "velocity": 20.0, "like_rate": 0.9})

    assert 0.0 <= low <= 1.0
    assert 0.0 <= high <= 1.0
    assert high >= low

