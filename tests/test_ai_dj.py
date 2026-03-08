from mantra.ai_dj import AIDJ


def test_ai_dj_generates_session_segments():
    dj = AIDJ()
    session = dj.generate_session(
        user_id="u1",
        recommendations=[
            {"track_id": "t1", "score": 0.9},
            {"track_id": "t2", "score": 0.8},
        ],
        persona="chill",
    )
    assert session["user_id"] == "u1"
    assert len(session["segments"]) == 2
    assert "narration" in session["segments"][0]

