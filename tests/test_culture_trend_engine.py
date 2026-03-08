from mantra.culture_trend_engine import CultureTrendEngine


def test_culture_trend_engine_shift_and_evolution():
    engine = CultureTrendEngine()
    shifts = engine.detect_cultural_music_shifts(
        [{"genre": "house"}, {"genre": "house"}, {"genre": "ambient"}],
        top_k=2,
    )
    evolution = engine.track_genre_evolution()
    assert shifts[0]["genre"] == "house"
    assert "house" in evolution
