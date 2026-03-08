from mantra.user_digital_twin import UserDigitalTwin


def test_user_digital_twin_session_and_eval():
    twin = UserDigitalTwin()
    twin.simulate_user_taste("u1", {"ambient": 3, "house": 1})
    session = twin.simulate_listening_sessions(
        "u1",
        [{"track_id": "t1", "genre": "house"}, {"track_id": "t2", "genre": "ambient"}],
        steps=2,
    )
    assert session[0]["genre"] == "ambient"
    eval_result = twin.evaluate_recommendation_changes("u1", [session[0]], [session[1]])
    assert "delta" in eval_result
