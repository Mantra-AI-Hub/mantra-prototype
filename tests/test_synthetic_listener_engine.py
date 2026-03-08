from mantra.intelligence.synthetic_listener_engine import (
    SyntheticListenerEngine,
    ListenerProfile,
    ListeningSession,
    TrackEngagement,
)


def test_generate_listeners_deterministic():
    engine_a = SyntheticListenerEngine(seed=1)
    engine_b = SyntheticListenerEngine(seed=1)
    listeners_a = engine_a.generate_listeners(10)
    listeners_b = engine_b.generate_listeners(10)
    assert [l.taste_vector for l in listeners_a] == [l.taste_vector for l in listeners_b]


def test_simulate_session_and_engagement():
    engine = SyntheticListenerEngine(seed=2)
    listeners = engine.generate_listeners(1)
    track_pool = [
        {"track_id": "t1", "vector": [0.1] * 8, "quality_score": 0.8, "emotional_score": 0.7, "novelty_score": 0.4},
        {"track_id": "t2", "vector": [0.9] * 8, "quality_score": 0.6, "emotional_score": 0.5, "novelty_score": 0.9},
    ]
    session = engine.simulate_session(listeners[0], track_pool)
    assert isinstance(session, ListeningSession)
    assert session.listener_id == listeners[0].listener_id
    engagement = engine.compute_engagement()
    assert isinstance(engagement["t1"], TrackEngagement)
    assert 0.0 <= engagement["t1"].virality_score <= 1.0


def test_generate_listener_report_defaults():
    engine = SyntheticListenerEngine(seed=3)
    report = engine.generate_listener_report("missing")
    assert report["plays"] == 0
    assert report["virality_score"] == 0.0
