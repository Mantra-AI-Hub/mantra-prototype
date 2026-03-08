from mantra.discovery_feed_ranker import DiscoveryFeedRanker


def test_discovery_feed_ranker_combines_signals():
    ranker = DiscoveryFeedRanker()
    ranked = ranker.rank(
        user_id="u1",
        candidates=[
            {
                "track_id": "t1",
                "score": 0.8,
                "rank_score": 0.9,
                "rl_score": 0.1,
                "bandit_score": 0.5,
                "trending_score": 0.7,
                "engagement_score": 0.8,
            },
            {
                "track_id": "t2",
                "score": 0.7,
                "rank_score": 0.7,
                "rl_score": 0.0,
                "bandit_score": 0.4,
                "trending_score": 0.1,
                "engagement_score": 0.3,
            },
        ],
        cluster_pref_tracks={"t1"},
    )
    assert ranked[0]["track_id"] == "t1"
    assert "discovery_score" in ranked[0]

