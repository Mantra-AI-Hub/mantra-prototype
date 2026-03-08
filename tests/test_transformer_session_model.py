from mantra.transformer_session_model import TransformerSessionModel


def test_transformer_session_model_sequence_and_predict():
    model = TransformerSessionModel()
    events = [
        {"user_id": "u1", "track_id": "t1"},
        {"user_id": "u1", "track_id": "t2"},
        {"user_id": "u2", "track_id": "t1"},
        {"user_id": "u2", "track_id": "t3"},
    ]
    dataset = model.build_sequence_dataset(events)
    assert dataset

    stats = model.train_transformer_session_model(events)
    assert stats["samples"] >= 1

    preds = model.predict_next_tracks("u1", ["t1"], top_k=3)
    assert preds
    assert all(len(item) == 2 for item in preds)

