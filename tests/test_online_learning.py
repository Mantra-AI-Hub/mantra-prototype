from uuid import uuid4

from mantra.online_learning import OnlineLearning


def test_online_learning_record_and_update_models():
    db_path = f"test_online_learning_{uuid4().hex}.db"
    learner = OnlineLearning(db_path=db_path)

    learner.record_interaction("u1", "t1", "click")
    learner.record_interaction("u1", "t2", "like")
    learner.record_interaction("u2", "t1", "click")

    stats = learner.update_models()
    assert stats["total_interactions"] == 3
    assert stats["click_events"] == 2
    assert stats["like_events"] == 1
