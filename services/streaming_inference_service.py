"""Service wrapper for realtime streaming inference."""

from __future__ import annotations

from mantra.realtime_streaming_inference import RealtimeStreamingInference


class StreamingInferenceService:
    def __init__(self, event_stream, recompute_fn):
        self.engine = RealtimeStreamingInference(event_stream=event_stream, recompute_fn=recompute_fn)

    def start(self) -> None:
        self.engine.start()

    def status(self):
        return self.engine.status()

