from mantra.bandit_recommender import BanditRecommender


def test_bandit_recommender_scores_and_reward_update():
    bandit = BanditRecommender(policy="ucb")
    bandit.update_reward("u1", "t1", 1.0)
    bandit.update_reward("u1", "t1", 1.0)
    scored = bandit.score_candidates(
        user_id="u1",
        candidates=[{"track_id": "t1", "score": 0.8}, {"track_id": "t2", "score": 0.7}],
    )
    assert len(scored) == 2
    assert all("bandit_score" in item for item in scored)

