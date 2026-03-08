from mantra.ai_label_engine import AILabelEngine


def test_ai_label_engine_discovery_playlist_and_growth():
    engine = AILabelEngine()
    discovered = engine.discover_new_artists(
        [
            {"artist": "a1", "growth": 5.0},
            {"artist": "a2", "growth": 3.0},
        ],
        top_k=2,
    )
    assert discovered[0]["artist"] == "a1"
    playlist = engine.generate_promotional_playlists(["a1", "a2"], length=4)
    assert len(playlist["new_talent_showcase"]) == 4
    growth = engine.track_fan_growth("a1", 100)
    assert growth["fans"] == 100

