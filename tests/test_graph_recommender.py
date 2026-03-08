from mantra.graph_recommender import GraphRecommender


def test_graph_recommender_recommendations():
    recommender = GraphRecommender()
    interactions = [
        {"user_id": "u1", "track_id": "t1", "weight": 1.0},
        {"user_id": "u2", "track_id": "t1", "weight": 1.0},
        {"user_id": "u2", "track_id": "t2", "weight": 1.0},
    ]
    recommender.build_user_track_graph(interactions)

    results = recommender.recommend_from_graph("u1", 5)
    assert results
    assert results[0][0] == "t2"

