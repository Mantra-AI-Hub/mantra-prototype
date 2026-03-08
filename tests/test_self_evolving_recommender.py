from mantra.self_evolving_recommender import SelfEvolvingRecommender


def test_self_evolving_recommender_retrain_and_switch():
    reco = SelfEvolvingRecommender()

    def asc(candidates, _):
        return sorted(candidates, key=lambda x: x[1])

    reco.register_strategy("asc", asc)
    reco.monitor_model_performance(0.2)
    reco.monitor_model_performance(0.3)
    reco.monitor_model_performance(0.4)
    assert reco.should_retrain()
    assert reco.auto_retrain_models() is True
    assert reco.auto_switch_ranking_strategies("asc") == "asc"
    ranked = reco.rank([("a", 0.9), ("b", 0.1)])
    assert ranked[0][0] == "b"
