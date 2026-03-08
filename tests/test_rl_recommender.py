from mantra.rl_recommender import RLRecommender


def test_rl_recommender_updates_and_selects_best_with_zero_epsilon():
    recommender = RLRecommender(epsilon=0.0)
    recommender.update_reward("u1", "t2", 2.0)

    picked = recommender.choose_action(
        "u1",
        [
            {"track_id": "t1", "rank_score": 0.7},
            {"track_id": "t2", "rank_score": 0.6},
        ],
    )

    assert picked is not None
    assert picked["track_id"] == "t2"

