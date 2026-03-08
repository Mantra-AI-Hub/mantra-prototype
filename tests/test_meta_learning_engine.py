from mantra.meta_learning_engine import MetaLearningEngine


def test_meta_learning_engine_records_and_recommends():
    engine = MetaLearningEngine(store_path="test_meta_learning_engine.json")
    engine.store_experiment_results("p1", {"quality": 0.8, "engagement": 0.7, "latency": 0.2})
    engine.store_experiment_results("p2", {"quality": 0.7, "engagement": 0.6, "latency": 0.1})
    best = engine.learn_best_ranking_pipeline()
    assert best is not None
    assert engine.recommend_best_model() in {"p1", "p2"}
