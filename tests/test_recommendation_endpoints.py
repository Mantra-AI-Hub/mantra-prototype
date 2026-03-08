from uuid import uuid4

import mantra.interfaces.api.api_server as api_server
from mantra.feature_store import FeatureStore
from mantra.session_model import SessionModel
from mantra.vector_index_service import VectorIndexService


def test_recommendation_feature_and_index_endpoints():
    run_id = uuid4().hex
    api_server._feature_store = FeatureStore(db_path=f"test_api_features_{run_id}.db")
    api_server._session_model = SessionModel()
    api_server._vector_index_service = VectorIndexService(index_path=f"test_api_index_{run_id}")

    user_resp = api_server.features_user(
        api_server.UserFeatureRequest(user_id="u1", features={"embedding": [1.0, 0.0], "mood": "calm"})
    )
    track_resp = api_server.features_track(
        api_server.TrackFeatureRequest(track_id="t1", features={"embedding": [0.9, 0.1], "genre": "ambient"})
    )

    assert user_resp["features"]["mood"] == "calm"
    assert track_resp["features"]["genre"] == "ambient"

    api_server.index_add(api_server.IndexAddRequest(track_id="t1", vector=[1.0, 0.0, 0.0, 0.0]))
    search_resp = api_server.index_search(api_server.IndexSearchRequest(vector=[1.0, 0.0, 0.0, 0.0], k=1))
    assert search_resp["results"]
    assert search_resp["results"][0]["track_id"] == "t1"

    api_server.events(api_server.EventRequest(event={"user_id": "u1", "track_id": "t1", "event": "play"}))
    # Event processing can be async; session endpoint should still be callable.
    session_resp = api_server.recommendations_session(user_id="u1")
    assert "results" in session_resp

    home_resp = api_server.recommendations_home(user_id="u1", k=5)
    assert "results" in home_resp
