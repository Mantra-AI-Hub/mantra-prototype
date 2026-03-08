import time
from uuid import uuid4

import mantra.interfaces.api.api_server as api_server
from mantra.embedding_trainer import EmbeddingTrainer
from mantra.feature_store import FeatureStore
from mantra.graph_recommender import GraphRecommender
from mantra.online_learning import OnlineLearning
from mantra.rl_recommender import RLRecommender
from mantra.session_model import SessionModel
from mantra.trend_detector import TrendDetector
from mantra.vector_index_service import VectorIndexService
from mantra.viral_predictor import ViralPredictor


def test_phase5155_recommendation_endpoints():
    run_id = uuid4().hex
    api_server._feature_store = FeatureStore(db_path=f"test_phase5155_features_{run_id}.db")
    api_server._session_model = SessionModel()
    api_server._vector_index_service = VectorIndexService(index_path=f"test_phase5155_index_{run_id}")
    api_server._online_learning = OnlineLearning(db_path=f"test_phase5155_online_{run_id}.db")
    api_server._graph_recommender = GraphRecommender()
    api_server._trend_detector = TrendDetector()
    api_server._viral_predictor = ViralPredictor()
    api_server._rl_recommender = RLRecommender(epsilon=0.0)
    api_server._embedding_trainer = EmbeddingTrainer(output_path=f"test_phase5155_embed_{run_id}.npy", dim=8)

    api_server.features_user(api_server.UserFeatureRequest(user_id="u1", features={"embedding": [1.0, 0.0]}))
    api_server.features_track(api_server.TrackFeatureRequest(track_id="t1", features={"embedding": [1.0, 0.0]}))
    api_server.features_track(api_server.TrackFeatureRequest(track_id="t2", features={"embedding": [0.0, 1.0]}))
    api_server.index_add(api_server.IndexAddRequest(track_id="t1", vector=[1.0, 0.0]))
    api_server.index_add(api_server.IndexAddRequest(track_id="t2", vector=[0.0, 1.0]))

    api_server.events(api_server.EventRequest(event={"user_id": "u1", "track_id": "t1", "event": "play"}))
    api_server.events(api_server.EventRequest(event={"user_id": "u2", "track_id": "t1", "event": "play"}))
    api_server.events(api_server.EventRequest(event={"user_id": "u2", "track_id": "t2", "event": "click"}))
    time.sleep(0.2)

    graph_resp = api_server.recommendations_graph(user_id="u1", k=5)
    assert "results" in graph_resp
    assert any(item["track_id"] == "t2" for item in graph_resp["results"])

    trending_resp = api_server.tracks_trending(top_k=5)
    assert trending_resp["results"]

    viral_resp = api_server.tracks_viral(window_hours=24, top_k=5)
    assert "results" in viral_resp

    rl_resp = api_server.rl_reward(api_server.RLRewardRequest(user_id="u1", track_id="t1", reward=1.5))
    assert rl_resp["track_id"] == "t1"

    home_resp = api_server.recommendations_home(user_id="u1", k=5)
    assert "results" in home_resp

    train_resp = api_server.embeddings_train(api_server.EmbeddingTrainRequest(incremental=False, save=False))
    assert train_resp["tracks"] >= 1

