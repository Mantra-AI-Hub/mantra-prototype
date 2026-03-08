from mantra.event_stream import EventStream
from mantra.realtime_streaming_inference import RealtimeStreamingInference


def test_realtime_streaming_inference_tracks_sessions():
    stream = EventStream()

    def recompute(user_id: str):
        return [{"track_id": f"{user_id}-rec", "score": 1.0}]

    engine = RealtimeStreamingInference(event_stream=stream, recompute_fn=recompute)
    engine.start()

    event = {"user_id": "u1", "track_id": "t1", "event": "play"}
    engine._on_event(event)  # direct invocation for deterministic unit test

    status = engine.status()
    assert status["active"] is True
    assert status["sessions"]["u1"] == 1
    assert engine.latest_recommendations["u1"][0]["track_id"] == "u1-rec"

