from datetime import datetime, timedelta, timezone

from mantra.trend_detector import TrendDetector


def test_trend_detector_trending_and_viral():
    detector = TrendDetector()
    now = datetime.now(timezone.utc)

    detector.ingest_event({"track_id": "t1", "timestamp": (now - timedelta(minutes=10)).isoformat()})
    detector.ingest_event({"track_id": "t1", "timestamp": (now - timedelta(minutes=5)).isoformat()})
    detector.ingest_event({"track_id": "t2", "timestamp": (now - timedelta(hours=2)).isoformat()})

    trending = detector.detect_trending_tracks(top_k=2)
    assert trending[0]["track_id"] == "t1"
    assert trending[0]["count"] >= 2

    velocity = detector.compute_velocity("t1")
    assert isinstance(velocity, float)

    viral = detector.detect_viral_tracks(window_hours=24, top_k=5)
    assert any(item["track_id"] == "t1" for item in viral)

