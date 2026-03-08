from mantra.reranker import rerank_results


def test_reranker_combines_similarity_metadata_popularity():
    results = [
        {
            "track_id": "a",
            "score": 0.8,
            "metadata": {"genre": "ambient", "fingerprint_hash_count": 400},
        },
        {
            "track_id": "b",
            "score": 0.81,
            "metadata": {"genre": "rock", "fingerprint_hash_count": 10},
        },
    ]

    reranked = rerank_results(results, filters={"genre": "ambient"})

    assert reranked
    assert reranked[0]["track_id"] == "a"
    assert "final_score" in reranked[0]
