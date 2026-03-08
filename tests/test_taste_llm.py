from mantra.taste_llm import TasteLLM


def test_taste_llm_summary_prediction_explanation():
    llm = TasteLLM()
    llm.ingest_events(
        "u1",
        [
            {"track_id": "t1", "artist": "a1", "genre": "ambient"},
            {"track_id": "t2", "artist": "a1", "genre": "ambient"},
            {"track_id": "t3", "artist": "a2", "genre": "lofi"},
        ],
    )
    summary = llm.generate_taste_summary("u1")
    assert "User prefers" in summary
    preds = llm.predict_next_genres_artists("u1")
    assert preds["genres"]
    explanation = llm.explain_recommendations("u1", [{"track_id": "t1"}])
    assert "Recommended next" in explanation

