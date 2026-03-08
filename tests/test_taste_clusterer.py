from mantra.taste_clusterer import TasteClusterer


def test_taste_clusterer_cluster_assign_and_recommend():
    clusterer = TasteClusterer(n_clusters=2)
    clusters = clusterer.cluster_users(
        {
            "u1": [1.0, 0.0],
            "u2": [0.9, 0.1],
            "u3": [0.0, 1.0],
        }
    )
    assert clusters
    cid = clusterer.assign_user_cluster("u1")
    assert isinstance(cid, int)

    clusterer.record_interaction("u1", "t1")
    clusterer.record_interaction("u1", "t1")
    clusterer.record_interaction("u2", "t2")

    recs = clusterer.recommend_from_cluster(cid, k=5)
    assert isinstance(recs, list)

