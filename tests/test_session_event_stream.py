from mantra.event_stream import EventStream
from mantra.session_model import SessionModel
import time


def test_event_stream_and_session_model_flow():
    stream = EventStream()
    session = SessionModel()

    def handler(event):
        uid = str(event.get("user_id"))
        tid = str(event.get("track_id"))
        if uid not in session.sessions:
            session.start_session(uid)
        session.update_session(uid, tid)

    stream.register_handler(handler)
    stream.start()
    stream.publish_event({"user_id": "u1", "track_id": "t1", "event": "play"})
    stream.publish_event({"user_id": "u1", "track_id": "t2", "event": "play"})
    time.sleep(0.2)
    stream.stop()

    recs = session.recommend("u1")
    assert recs
    assert recs[0] == "t2"
