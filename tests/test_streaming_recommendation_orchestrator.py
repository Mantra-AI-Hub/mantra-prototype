from mantra.streaming_recommendation_orchestrator import StreamingRecommendationOrchestrator


def test_streaming_recommendation_orchestrator_choose_and_status():
    orchestrator = StreamingRecommendationOrchestrator()
    best = orchestrator.choose_best_engine({})
    assert best in {"vector", "graph", "gnn", "bandit", "transformer"}
    cold = orchestrator.choose_best_engine({"cold_start": True})
    assert cold == "bandit"
    orchestrator.update_engine_health("gnn", 0.95)
    status = orchestrator.status()
    assert "engine_health" in status

