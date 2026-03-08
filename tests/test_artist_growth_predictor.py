from mantra.artist_growth_predictor import ArtistGrowthPredictor


def test_artist_growth_predictor_breakout():
    predictor = ArtistGrowthPredictor()
    predictor.ingest_events(
        [
            {"artist": "a1", "event": "play", "user_id": "u1"},
            {"artist": "a1", "event": "like", "user_id": "u2"},
            {"artist": "a2", "event": "play", "user_id": "u1"},
        ]
    )
    growth = predictor.predict_artist_popularity_growth("a1")
    assert growth > 0
    breakout = predictor.predict_breakout_artists(top_k=2)
    assert breakout
    audience = predictor.analyze_audience_expansion("a1")
    assert audience["unique_listeners"] >= 1

