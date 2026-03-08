import mantra.interfaces.api.api_server as api_server
from mantra.artist_intelligence import ArtistIntelligence
from mantra.distributed_embedding_pipeline import DistributedEmbeddingPipeline
from mantra.embedding_trainer import EmbeddingTrainer
from mantra.feature_store import FeatureStore
from mantra.graph_recommender import GraphRecommender
from mantra.online_learning import OnlineLearning
from mantra.realtime_recommender import RealtimeRecommender
from mantra.rl_recommender import RLRecommender
from mantra.session_model import SessionModel
from mantra.taste_clusterer import TasteClusterer
from mantra.transformer_session_model import TransformerSessionModel
from mantra.trend_detector import TrendDetector
from mantra.vector_index_service import VectorIndexService
from mantra.viral_predictor import ViralPredictor


def test_phase5660_api_endpoints():
    api_server._feature_store = FeatureStore(db_path="test_phase5660_features.db")
    api_server._session_model = SessionModel()
    api_server._vector_index_service = VectorIndexService(index_path="test_phase5660_index")
    api_server._online_learning = OnlineLearning(db_path="test_phase5660_online.db")
    api_server._graph_recommender = GraphRecommender()
    api_server._trend_detector = TrendDetector()
    api_server._viral_predictor = ViralPredictor()
    api_server._rl_recommender = RLRecommender(epsilon=0.0)
    api_server._embedding_trainer = EmbeddingTrainer(output_path="test_phase5660_embeds.npy", dim=8)
    api_server._transformer_session_model = TransformerSessionModel()
    api_server._taste_clusterer = TasteClusterer(n_clusters=2)
    api_server._artist_intelligence = ArtistIntelligence()
    api_server._distributed_embedding_pipeline = DistributedEmbeddingPipeline(api_server._embedding_trainer)
    api_server._realtime_recommender = RealtimeRecommender(
        vector_index_service=api_server._vector_index_service,
        graph_recommender=api_server._graph_recommender,
        transformer_session_model=api_server._transformer_session_model,
        trend_detector=api_server._trend_detector,
    )

    api_server.features_user(api_server.UserFeatureRequest(user_id="u1", features={"embedding": [1.0, 0.0]}))
    api_server.features_user(api_server.UserFeatureRequest(user_id="u2", features={"embedding": [0.9, 0.1]}))
    api_server.features_track(api_server.TrackFeatureRequest(track_id="t1", features={"embedding": [1.0, 0.0]}))
    api_server.features_track(api_server.TrackFeatureRequest(track_id="t2", features={"embedding": [0.0, 1.0]}))
    api_server.index_add(api_server.IndexAddRequest(track_id="t1", vector=[1.0, 0.0]))
    api_server.index_add(api_server.IndexAddRequest(track_id="t2", vector=[0.0, 1.0]))

    api_server._track_store.add_track(
        {
            "track_id": "t1",
            "filename": "t1.wav",
            "duration": 1.0,
            "embedding_path": "x",
            "fingerprint_hash_count": 0,
            "artist": "a1",
            "genre": "ambient",
            "year": 2020,
            "tags": [],
        }
    )
    api_server._track_store.add_track(
        {
            "track_id": "t2",
            "filename": "t2.wav",
            "duration": 1.0,
            "embedding_path": "x",
            "fingerprint_hash_count": 0,
            "artist": "a2",
            "genre": "ambient",
            "year": 2021,
            "tags": [],
        }
    )

    api_server.events(api_server.EventRequest(event={"user_id": "u1", "track_id": "t1", "event": "play"}))
    api_server.events(api_server.EventRequest(event={"user_id": "u2", "track_id": "t1", "event": "play"}))
    api_server.events(api_server.EventRequest(event={"user_id": "u2", "track_id": "t2", "event": "click"}))

    realtime_resp = api_server.recommendations_realtime(user_id="u1", k=5)
    assert "results" in realtime_resp

    cluster_resp = api_server.users_cluster(user_id="u1")
    assert "cluster_id" in cluster_resp

    artist_resp = api_server.artists_similar(artist_id="a1", user_id=None, k=5)
    assert "results" in artist_resp

    batch_resp = api_server.embeddings_batch_train(api_server.BatchEmbeddingTrainRequest(save=True, shards=2))
    assert "training" in batch_resp
    assert "shards" in batch_resp

    stats = api_server.recommendation_stats()
    assert stats["interactions"] >= 1

