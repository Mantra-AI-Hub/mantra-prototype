from mantra.model_evolution_manager import ModelEvolutionManager


def test_model_evolution_manager_versions_and_prunes():
    manager = ModelEvolutionManager()
    manager.version_model("rank", "v1", "models/test_rank_v1.bin", score=0.2)
    manager.version_model("rank", "v2", "models/test_rank_v2.bin", score=0.9)
    removed = manager.prune_weak_models(threshold=0.5)
    assert "rank:v1" in removed
