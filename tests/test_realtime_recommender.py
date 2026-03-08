from mantra.graph_recommender import GraphRecommender
from mantra.realtime_recommender import RealtimeRecommender
from mantra.transformer_session_model import TransformerSessionModel
from mantra.trend_detector import TrendDetector
from mantra.vector_index_service import VectorIndexService


def test_realtime_recommender_generates_candidates():
    vector_service = VectorIndexService(index_path="test_realtime_index")
    vector_service.add_vector("t1", [1.0, 0.0])
    vector_service.add_vector("t2", [0.0, 1.0])

    graph = GraphRecommender()
    graph.build_user_track_graph(
        [
            {"user_id": "u1", "track_id": "t1", "weight": 1.0},
            {"user_id": "u2", "track_id": "t1", "weight": 1.0},
            {"user_id": "u2", "track_id": "t2", "weight": 1.0},
        ]
    )

    transformer = TransformerSessionModel()
    transformer.train_transformer_session_model(
        [
            {"user_id": "u1", "track_id": "t1"},
            {"user_id": "u1", "track_id": "t2"},
        ]
    )
    trend = TrendDetector()
    trend.ingest_event({"track_id": "t1"})

    realtime = RealtimeRecommender(vector_service, graph, transformer, trend)
    scores = realtime.generate_candidates("u1", [1.0, 0.0], ["t1"], top_k=10)
    assert scores
    assert "t1" in scores or "t2" in scores

