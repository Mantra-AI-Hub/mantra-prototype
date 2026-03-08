from mantra.gnn_recommender import GNNRecommender


def test_gnn_recommender_build_train_recommend():
    gnn = GNNRecommender()
    gnn.build_user_track_graph(
        [
            {"user_id": "u1", "track_id": "t1", "weight": 1.0},
            {"user_id": "u2", "track_id": "t1", "weight": 1.0},
            {"user_id": "u2", "track_id": "t2", "weight": 1.0},
        ]
    )
    status = gnn.train_gnn_model()
    assert status["trained"] is True
    recs = gnn.recommend_with_gnn("u1", k=5)
    assert recs
    assert recs[0][0] == "t2"

