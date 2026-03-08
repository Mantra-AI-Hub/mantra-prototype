from mantra.collaborative_ai_band import CollaborativeAIBand


def test_collaborative_ai_band_creates_track():
    band = CollaborativeAIBand(agents=["a", "b"])
    output = band.create_track_collaboratively("test", rounds=2)
    assert len(output["contributions"]) == 4
