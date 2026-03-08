from mantra.global_trend_intelligence import GlobalTrendIntelligence


def test_global_trend_intelligence_pipeline():
    intel = GlobalTrendIntelligence()
    intel.ingest(
        [
            {"track_id": "t1", "genre": "ambient"},
            {"track_id": "t1", "genre": "ambient"},
            {"track_id": "t2", "genre": "house"},
        ]
    )
    trends = intel.detect_global_trends(top_k=2)
    assert trends[0]["track_id"] == "t1"
    clusters = intel.cluster_trending_tracks(top_k=5)
    assert isinstance(clusters, dict)
    emerging = intel.identify_emerging_genres(top_k=5)
    assert isinstance(emerging, list)
    growth = intel.predict_viral_growth("t1")
    assert isinstance(growth, float)

