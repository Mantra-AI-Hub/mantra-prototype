"""FastAPI server exposing MANTRA audio indexing and search endpoints."""

from __future__ import annotations

import io
import time
import uuid
import wave
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional, Tuple

import numpy as np
from mantra.fingerprint_engine import generate_fingerprints, generate_fingerprints_from_audio
from mantra.fingerprint_index import FingerprintIndex
from fastapi import Body, FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from mantra.cross_modal_search import search_by_lyrics, search_by_text
from mantra.hybrid_search import hybrid_search
from mantra.experiment_router import ExperimentRouter
from mantra.event_stream import EventStream
from mantra.artist_intelligence import ArtistIntelligence
from mantra.artist_growth_predictor import ArtistGrowthPredictor
from mantra.autonomous_optimizer import AutonomousOptimizer
from mantra.bandit_recommender import BanditRecommender
from mantra.discovery_feed_ranker import DiscoveryFeedRanker
from mantra.distributed_embedding_pipeline import DistributedEmbeddingPipeline
from mantra.embedding_trainer import EmbeddingTrainer
from mantra.feature_store import FeatureStore
from mantra.foundation_music_embeddings import FoundationMusicEmbeddings
from mantra.generation_orchestrator import GenerationOrchestrator
from mantra.gnn_recommender import GNNRecommender
from mantra.global_trend_intelligence import GlobalTrendIntelligence
from mantra.graph_recommender import GraphRecommender
from mantra.knowledge_graph_builder import KnowledgeGraphBuilder
from mantra.music_agent_system import MusicAgentSystem
from mantra.music_foundation_model import MusicFoundationModel
from mantra.music_generation_pipeline import MusicGenerationPipeline
from mantra.online_learning import OnlineLearning
from mantra.ranker import rank_tracks
from pydantic import BaseModel
from mantra.music_assistant import handle_query
from mantra.playlist_generator import generate_playlist
from mantra.rl_recommender import RLRecommender
from mantra.realtime_recommender import RealtimeRecommender
from mantra.realtime_feature_aggregator import RealtimeFeatureAggregator
from mantra.realtime_streaming_inference import RealtimeStreamingInference
from mantra.session_model import SessionModel
from mantra.services.ai_dj_service import AIDJService
from mantra.services.generation_service import GenerationService
from mantra.services.streaming_inference_service import StreamingInferenceService
from mantra.ecosystem.virtual_artist_factory import VirtualArtistFactory
from mantra.generation.ai_producer_engine import AIProducerEngine
from mantra.stream_recognition import StreamRecognizer
from mantra.streaming_recommendation_orchestrator import StreamingRecommendationOrchestrator
from mantra.taste_llm import TasteLLM
from mantra.taste_clusterer import TasteClusterer
from mantra.transformer_session_model import TransformerSessionModel
from mantra.trend_detector import TrendDetector
from mantra.vector_index_service import VectorIndexService
from mantra.viral_predictor import ViralPredictor
from mantra.ai_label_engine import AILabelEngine
from mantra.adaptive_style_engine import AdaptiveStyleEngine
from mantra.ai_music_supervisor import AIMusicSupervisor
from mantra.generation.ai_producer_engine import AIProducerEngine
from mantra.auto_scaling_ai import AutoScalingAI
from mantra.autonomous_experimentation import AutonomousExperimentation
from mantra.collaborative_ai_band import CollaborativeAIBand
from mantra.culture_trend_engine import CultureTrendEngine
from mantra.distributed_training_orchestrator import DistributedTrainingOrchestrator
from mantra.ecosystem_simulator import EcosystemSimulator
from mantra.fanbase_model import FanbaseModel
from mantra.generative_music_lab import GenerativeMusicLab
from mantra.global_music_map import GlobalMusicMap
from mantra.market_simulator import MarketSimulator
from mantra.meta_learning_engine import MetaLearningEngine
from mantra.model_evolution_manager import ModelEvolutionManager
from mantra.promotion_optimizer import PromotionOptimizer
from mantra.self_evolving_recommender import SelfEvolvingRecommender
from mantra.self_improvement_loop import SelfImprovementLoop
from mantra.user_digital_twin import UserDigitalTwin
from mantra.ecosystem.virtual_artist_factory import VirtualArtistFactory
from mantra.intelligence.global_music_intelligence import (
    GlobalMusicIntelligenceEngine,
    HitPrediction,
    MusicFeatureVector,
)
from mantra.intelligence.music_foundation_model import MusicFoundationModel as IntelligenceMusicFoundationModel
from mantra.intelligence.music_genome_engine import MusicGenomeEngine
from mantra.intelligence.music_genome_store import MusicGenomeStore

from mantra.audio_engine.feature_extractor import (
    extract_chroma,
    extract_pitch_contour,
    extract_spectral_features,
    extract_tempo,
)
from mantra.database.track_store import TrackStore
from mantra.dataset_manager import DatasetManager
from mantra.embedding_engine import EmbeddingEngine
from mantra.fingerprint_engine.audio_fingerprint import generate_fingerprint, match_fingerprint
from mantra.index_updater import IndexUpdater
from mantra.ingestion.dataset_ingestor import enqueue_all_audio_files
from mantra.ingestion.progress_tracker import IngestionProgressTracker
from mantra.metrics import metrics_registry
from mantra.pipeline.job_queue import JobQueue
from mantra.pipeline.worker import BackgroundWorker
from mantra.recommendation_engine import RecommendationEngine
from mantra.search_engine import search_cache_metrics, semantic_search, semantic_search_batch
from mantra.vector_engine.embedding_builder import EMBEDDING_SIZE, build_music_embedding
from mantra.vector_index.faiss_index import VectorIndex


AudioData = Tuple[np.ndarray, int]

app = FastAPI(title="MANTRA API", version="0.1.0")

_DATA_DIR = Path("data")
_DATA_DIR.mkdir(parents=True, exist_ok=True)

_TRACK_DB_PATH = str(_DATA_DIR / "mantra_tracks.db")
_VECTOR_INDEX_PATH = str(_DATA_DIR / "mantra_vector_index")
_WEB_DIR = Path("web")

_track_store = TrackStore(db_path=_TRACK_DB_PATH)
_dataset_manager = DatasetManager(db_path=_TRACK_DB_PATH)
_embedding_engine = EmbeddingEngine()
_embedding_engine.load_model()
_experiment_router = ExperimentRouter(db_path=str(_DATA_DIR / "experiments.db"))
_feature_store = FeatureStore(db_path=str(_DATA_DIR / "feature_store.db"))
_vector_index_service = VectorIndexService(index_path=str(_DATA_DIR / "home_vector_index"))
_session_model = SessionModel()
_event_stream = EventStream()
_online_learning = OnlineLearning(db_path=str(_DATA_DIR / "online_learning.db"))
_graph_recommender = GraphRecommender()
_trend_detector = TrendDetector()
_viral_predictor = ViralPredictor()
_rl_recommender = RLRecommender(epsilon=0.0)
_embedding_trainer = EmbeddingTrainer(output_path=str(_DATA_DIR / "track_embeddings.npy"))
_transformer_session_model = TransformerSessionModel()
_taste_clusterer = TasteClusterer(n_clusters=4)
_artist_intelligence = ArtistIntelligence()
_distributed_embedding_pipeline = DistributedEmbeddingPipeline(embedding_trainer=_embedding_trainer)
_foundation_embeddings = FoundationMusicEmbeddings(feature_store=_feature_store)
_gnn_recommender = GNNRecommender()
_bandit_recommender = BanditRecommender(policy="thompson")
_realtime_feature_aggregator = RealtimeFeatureAggregator(window_minutes=60)
_discovery_feed_ranker = DiscoveryFeedRanker()
_music_generation_pipeline = MusicGenerationPipeline()
_music_foundation_model = MusicFoundationModel()
_taste_llm = TasteLLM()
_autonomous_optimizer = AutonomousOptimizer()
_global_trend_intel = GlobalTrendIntelligence()
_artist_growth_predictor = ArtistGrowthPredictor()
_music_agent_system = MusicAgentSystem()
_generation_orchestrator = GenerationOrchestrator(_music_generation_pipeline)
_streaming_reco_orchestrator = StreamingRecommendationOrchestrator()
_knowledge_graph_builder = KnowledgeGraphBuilder()
_ai_label_engine = AILabelEngine()
_self_evolving_recommender = SelfEvolvingRecommender()
_meta_learning_engine = MetaLearningEngine()
_autonomous_experimentation = AutonomousExperimentation()
_user_digital_twin = UserDigitalTwin()
_ecosystem_simulator = EcosystemSimulator()
_generative_music_lab = GenerativeMusicLab()
_ai_producer_engine = AIProducerEngine()
_collaborative_ai_band = CollaborativeAIBand()
_adaptive_style_engine = AdaptiveStyleEngine()
_virtual_artist_factory = VirtualArtistFactory()
_fanbase_model = FanbaseModel()
_promotion_optimizer = PromotionOptimizer()
_culture_trend_engine = CultureTrendEngine()
_market_simulator = MarketSimulator()
_global_music_map = GlobalMusicMap()
_auto_scaling_ai = AutoScalingAI()
_distributed_training_orchestrator = DistributedTrainingOrchestrator()
_model_evolution_manager = ModelEvolutionManager()
_intelligence_foundation_model = IntelligenceMusicFoundationModel()
_music_genome_engine = MusicGenomeEngine()
_music_genome_store = MusicGenomeStore(genome_engine=_music_genome_engine)

_global_music_intelligence = GlobalMusicIntelligenceEngine(feature_store=_feature_store)
_ai_music_supervisor = AIMusicSupervisor(
    recommender=_self_evolving_recommender,
    experimentation=_autonomous_experimentation,
    foundation_model=_intelligence_foundation_model,
    genome_engine=_music_genome_engine,
    genome_store=_music_genome_store,
)
_self_improvement_loop = SelfImprovementLoop(
    supervisor=_ai_music_supervisor,
    evolution_manager=_model_evolution_manager,
)
_realtime_recommender = RealtimeRecommender(
    vector_index_service=_vector_index_service,
    graph_recommender=_graph_recommender,
    transformer_session_model=_transformer_session_model,
    trend_detector=_trend_detector,
    genome_store=_music_genome_store,
    foundation_model=_intelligence_foundation_model,
)
_ai_dj_service = AIDJService()
_generation_service = GenerationService()
try:
    _vector_index = VectorIndex.load(_VECTOR_INDEX_PATH)
except FileNotFoundError:
    _vector_index = VectorIndex(dimension=EMBEDDING_SIZE)
_fingerprint_db: Dict[str, List[Tuple[str, int]]] = _track_store.list_fingerprints()
_job_queue = JobQueue()
_worker = BackgroundWorker(
    queue=_job_queue,
    vector_index=_vector_index,
    track_store=_track_store,
    fingerprint_db=_fingerprint_db,
    vector_index_path=_VECTOR_INDEX_PATH,
)
_ingestion_tracker = IngestionProgressTracker()
_index_updater = IndexUpdater(
    vector_index=_vector_index,
    vector_index_path=_VECTOR_INDEX_PATH,
    track_store=_track_store,
    feature_store=_feature_store,
)
_recommendation_engine = RecommendationEngine(
    track_store=_track_store,
    genome_store=_music_genome_store,
    foundation_model=_intelligence_foundation_model,
)
_fingerprint_index_path = str(_DATA_DIR / "fingerprint_index.json")
try:
    _fingerprint_index = FingerprintIndex.load(_fingerprint_index_path)
except FileNotFoundError:
    _fingerprint_index = FingerprintIndex()
_stream_recognizer = StreamRecognizer(fingerprint_index=_fingerprint_index, sample_rate=22050)
_index_lock = Lock()


def _stream_recompute(user_id: str) -> List[Dict[str, object]]:
    response = recommendations_realtime(user_id=user_id, k=10)
    return list(response.get("results", []))


_streaming_inference = RealtimeStreamingInference(event_stream=_event_stream, recompute_fn=_stream_recompute)
_streaming_service = StreamingInferenceService(event_stream=_event_stream, recompute_fn=_stream_recompute)
_streaming_inference.start()
_streaming_service.start()

if _WEB_DIR.exists():
    app.mount("/web/static", StaticFiles(directory=str(_WEB_DIR)), name="web-static")


def _event_handler(event: Dict[str, object]) -> None:
    user_id = str(event.get("user_id") or "")
    track_id = str(event.get("track_id") or "")
    event_type = str(event.get("event") or "")
    if "timestamp" not in event:
        event["timestamp"] = datetime.now(timezone.utc).isoformat()

    # Keep artist graph current for artist intelligence lookups.
    _artist_intelligence.build_artist_graph(_track_store.list_tracks())
    _trend_detector.ingest_event(event)
    _realtime_feature_aggregator.update_from_event(event)
    _global_trend_intel.ingest([event])

    if track_id:
        track_features = _feature_store.get_track_features(track_id) or {}
        play_count = int(track_features.get("play_count", 0))
        like_count = int(track_features.get("like_count", 0))
        if event_type in {"play", "click", "like"}:
            play_count += 1
        if event_type == "like":
            like_count += 1
        like_rate = float(like_count / max(1, play_count))
        velocity = _trend_detector.compute_velocity(track_id)
        track_features.update(
            {
                "play_count": play_count,
                "like_count": like_count,
                "like_rate": like_rate,
                "velocity": float(velocity),
            }
        )
        _feature_store.store_track_features(track_id, track_features)
        rt_feats = _realtime_feature_aggregator.compute_features(track_id)
        track_features.update(rt_feats)
        _feature_store.store_track_features(track_id, track_features)
        metadata = _track_store.get_track(track_id) or {}
        mm = _music_foundation_model.build_multimodal_embedding(
            audio=track_id,
            lyrics=str(metadata.get("lyrics") or ""),
            metadata=metadata,
            artist_id=str(metadata.get("artist") or ""),
        )
        _foundation_embeddings.cache_embedding(track_id, mm.tolist())
        _knowledge_graph_builder.build_from_records(
            [
                {
                    "user_id": user_id,
                    "track_id": track_id,
                    "artist": metadata.get("artist"),
                    "genre": metadata.get("genre"),
                    "playlist_id": "",
                }
            ]
        )

    if not user_id:
        return
    if user_id not in _session_model.sessions:
        _session_model.start_session(user_id)
    if track_id and event_type in {"play", "click", "like"}:
        _session_model.update_session(user_id, track_id)
        _online_learning.record_interaction(user_id, track_id, event_type)
        _transformer_session_model.ingest_event(user_id=user_id, track_id=track_id)
        reward = 1.0 if event_type == "play" else (1.5 if event_type == "click" else 2.0)
        _bandit_recommender.update_reward(user_id, track_id, reward)
        _embedding_trainer.update_embeddings_incrementally([{"track_id": track_id, "reward": reward}])
        _distributed_embedding_pipeline.incremental_update_embeddings([{"track_id": track_id, "reward": reward}])
        _taste_clusterer.record_interaction(user_id, track_id)
        metadata = _track_store.get_track(track_id) or {}
        _artist_intelligence.record_user_artist(user_id, str(metadata.get("artist") or ""))
        _artist_growth_predictor.ingest_events(
            [
                {
                    "user_id": user_id,
                    "track_id": track_id,
                    "artist": metadata.get("artist"),
                    "genre": metadata.get("genre"),
                    "event": event_type,
                }
            ]
        )
        _taste_llm.ingest_events(
            user_id=user_id,
            events=[
                {
                    "track_id": track_id,
                    "artist": metadata.get("artist"),
                    "genre": metadata.get("genre"),
                    "event": event_type,
                }
            ],
        )


def _refresh_user_clusters() -> Dict[str, int]:
    user_embeddings: Dict[str, List[float]] = {}
    interactions = _online_learning.list_interactions()
    known_users = {str(item.get("user_id")) for item in interactions if item.get("user_id")}
    for uid in known_users:
        features = _feature_store.get_user_features(uid) or {}
        embedding = features.get("embedding")
        if isinstance(embedding, list) and embedding:
            user_embeddings[uid] = [float(v) for v in embedding]
    return _taste_clusterer.cluster_users(user_embeddings)


def _rebuild_graph_from_interactions() -> None:
    interactions = _online_learning.list_interactions()
    weighted = []
    for item in interactions:
        event_type = str(item.get("event") or "")
        reward = 1.0 if event_type == "play" else (1.5 if event_type == "click" else 2.0)
        weighted.append(
            {
                "user_id": item.get("user_id"),
                "track_id": item.get("track_id"),
                "weight": reward,
            }
        )
    _graph_recommender.build_user_track_graph(weighted)
    _gnn_recommender.build_user_track_graph(weighted)
    _gnn_recommender.train_gnn_model()


_event_stream.register_handler(_event_handler)
_event_stream.start()


def _queue_depth() -> int:
    if hasattr(_job_queue, "_items"):
        return int(len(getattr(_job_queue, "_items")))
    if hasattr(_job_queue, "_queue"):
        return int(len(getattr(_job_queue, "_queue")))
    return 0


def _sync_fingerprint_index() -> None:
    global _fingerprint_index, _stream_recognizer
    rebuilt = FingerprintIndex()
    for track_id, fps in _fingerprint_db.items():
        rebuilt.add(track_id, fps)
    _fingerprint_index = rebuilt
    _fingerprint_index.save(_fingerprint_index_path)
    _stream_recognizer = StreamRecognizer(fingerprint_index=_fingerprint_index, sample_rate=22050)


class SearchRequest(BaseModel):
    query_audio_path: str
    top_k: int = 5
    filters: Dict[str, object] | None = None


class BatchSearchRequest(BaseModel):
    queries: List[Dict[str, object]]
    top_k: int = 5


class DatasetRegisterRequest(BaseModel):
    dataset_id: str
    name: str


class DatasetIngestRequest(BaseModel):
    dataset_id: str
    path: str


class ExperimentCreateRequest(BaseModel):
    experiment_id: str
    name: str
    model_a: str
    model_b: str


class AddTrackRequest(BaseModel):
    audio_path: str
    metadata: Dict[str, object] = {}


class RecognizeRequest(BaseModel):
    audio_path: str


class StreamRecognizeRequest(BaseModel):
    samples: List[float]
    sample_rate: int = 22050


class TextSearchRequest(BaseModel):
    query: str
    top_k: int = 5


class AssistantQueryRequest(BaseModel):
    prompt: str


class EventRequest(BaseModel):
    event: Dict[str, object]


class UserFeatureRequest(BaseModel):
    user_id: str
    features: Dict[str, object]


class TrackFeatureRequest(BaseModel):
    track_id: str
    features: Dict[str, object]


class IndexAddRequest(BaseModel):
    track_id: str
    vector: List[float]


class IndexSearchRequest(BaseModel):
    vector: List[float]
    k: int = 10


class RLRewardRequest(BaseModel):
    user_id: str
    track_id: str
    reward: float


class EmbeddingTrainRequest(BaseModel):
    incremental: bool = False
    save: bool = True


class BatchEmbeddingTrainRequest(BaseModel):
    save: bool = True
    shards: int = 4


class BanditRewardRequest(BaseModel):
    user_id: str
    track_id: str
    reward: float


class MusicGenerateRequest(BaseModel):
    prompt: str
    seconds: int = 5


class StyleTransferRequest(BaseModel):
    track: str
    style: str


class RemixRequest(BaseModel):
    track: str
    style: str = "electronic"


class AdvancedGenerateRequest(BaseModel):
    prompt: str
    mode: str = "musicgen"
    seconds: int = 5


class CreateArtistRequest(BaseModel):
    name: str | None = None
    genre: str | None = None
    persona: str | None = None


class ProduceMusicRequest(BaseModel):
    genre: str | None = None
    mood: str | None = None
    tempo: int | None = None
    genome: Dict[str, object] | None = None


class AIMusicCollaborateRequest(BaseModel):
    prompt: str
    rounds: int = 2


class AnalyzeMusicRequest(BaseModel):
    audio_path: str | None = None
    midi_path: str | None = None
    feature_vector: List[float] | None = None


class MusicGenomeRequest(BaseModel):
    audio_path: str


class GenerateFromGenomeRequest(BaseModel):
    genome: Dict[str, object]


class IntelligencePredictRequest(BaseModel):
    track_id: Optional[str] = None
    tempo: float
    energy: float
    mood: float
    novelty: float
    rhythm_complexity: float
    harmonic_density: float
    emotional_intensity: Optional[float] = None


def _decode_wav_bytes(raw_bytes: bytes) -> AudioData:
    if not raw_bytes:
        raise ValueError("Empty audio payload")

    with wave.open(io.BytesIO(raw_bytes), "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_rate = wav_file.getframerate()
        sample_width = wav_file.getsampwidth()
        frames = wav_file.readframes(wav_file.getnframes())

    if sample_width == 1:
        dtype = np.uint8
        offset = 128.0
        scale = 128.0
    elif sample_width == 2:
        dtype = np.int16
        offset = 0.0
        scale = 32768.0
    elif sample_width == 4:
        dtype = np.int32
        offset = 0.0
        scale = 2147483648.0
    else:
        raise ValueError(f"Unsupported WAV sample width: {sample_width} bytes")

    audio = np.frombuffer(frames, dtype=dtype).astype(np.float32)
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)

    audio = (audio - offset) / scale
    return audio.astype(np.float32, copy=False), int(sample_rate)


def _extract_feature_payload(audio: AudioData) -> Dict[str, object]:
    return {
        "chroma": extract_chroma(audio),
        "tempo": extract_tempo(audio),
        "pitch_contour": extract_pitch_contour(audio),
        "spectral": extract_spectral_features(audio),
    }


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/index/audio")
def index_audio(
    audio_bytes: bytes = Body(..., media_type="application/octet-stream"),
    filename: str = Query(default="audio.wav"),
    track_id: str | None = Query(default=None),
    artist: str | None = None,
    album: str | None = None,
    genre: str | None = None,
    tags: str | None = None,
    year: int | None = None,
) -> Dict[str, object]:
    if not filename.lower().endswith(".wav"):
        raise HTTPException(status_code=400, detail="Only WAV uploads are supported")

    try:
        start = time.perf_counter()
        audio = _decode_wav_bytes(audio_bytes)
        metrics_registry.embedding_latency = float(time.perf_counter() - start)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except wave.Error as exc:
        raise HTTPException(status_code=400, detail="Invalid WAV payload") from exc

    if track_id:
        resolved_track_id = str(track_id)
    else:
        candidate = Path(str(filename or "")).stem.strip()
        resolved_track_id = candidate or uuid.uuid4().hex

    duration = float(len(audio[0]) / max(1, int(audio[1])))
    created_at = datetime.now(timezone.utc).isoformat()

    parsed_tags = [value.strip() for value in str(tags or "").split(",") if value.strip()]
    structure = _intelligence_foundation_model.analyze_structure(filename)
    foundation_embedding = _intelligence_foundation_model.embed_features(
        [
            float(structure["tempo"]) / 200.0,
            float(sum(structure["spectral_features"]) / max(1, len(structure["spectral_features"]))),
            float(len(structure["rhythm_pattern"])) / 16.0,
        ]
    )
    genome = _music_genome_engine.extract_genome(filename)

    with _index_lock:
        _track_store.add_track(
            {
                "track_id": resolved_track_id,
                "filename": filename,
                "duration": duration,
                "embedding_path": _VECTOR_INDEX_PATH,
                "fingerprint_hash_count": 0,
                "created_at": created_at,
                "artist": artist or "",
                "album": album or "",
                "genre": genre or "",
                "tags": parsed_tags,
                "year": year,
            }
        )
        _music_genome_store.store_genome(resolved_track_id, genome)
        _feature_store.store_track_features(
            resolved_track_id,
            {
                "foundation_embedding": foundation_embedding.tolist(),
                "music_structure": structure,
                "music_genome": genome,
            },
        )
        _job_queue.enqueue(
            "index_audio",
            {
                "track_id": resolved_track_id,
                "filename": filename,
                "embedding_path": _VECTOR_INDEX_PATH,
                "audio_bytes": audio_bytes,
                "artist": artist or "",
                "album": album or "",
                "genre": genre or "",
                "tags": parsed_tags,
                "year": year,
            },
        )
        metrics_registry.queue_depth = _queue_depth()

    return {
        "track_id": resolved_track_id,
        "embedding_size": EMBEDDING_SIZE,
        "fingerprint_size": 0,
        "metadata": _track_store.get_track(resolved_track_id),
        "foundation_embedding_summary": {
            "dimension": int(foundation_embedding.shape[0]),
            "norm": float(np.linalg.norm(foundation_embedding)),
        },
        "music_genome": genome,
    }


@app.post("/search/audio")
def search_audio(
    audio_bytes: bytes = Body(..., media_type="application/octet-stream"),
    filename: str = Query(default="query.wav"),
    top_k: int = Query(default=5),
) -> Dict[str, object]:
    if not filename.lower().endswith(".wav"):
        raise HTTPException(status_code=400, detail="Only WAV uploads are supported")

    try:
        audio = _decode_wav_bytes(audio_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except wave.Error as exc:
        raise HTTPException(status_code=400, detail="Invalid WAV payload") from exc

    started = time.perf_counter()
    features = _extract_feature_payload(audio)
    embedding = build_music_embedding(features)

    with _index_lock:
        matches = _vector_index.search(embedding, top_k=top_k)

    response = {
        "results": [
            {"track_id": track, "score": score, "metadata": _track_store.get_track(track)}
            for track, score in matches
        ],
    }
    metrics_registry.search_latency = float(time.perf_counter() - started)
    return response


@app.post("/fingerprint/match")
def fingerprint_match(
    audio_bytes: bytes = Body(..., media_type="application/octet-stream"),
    filename: str = Query(default="snippet.wav"),
    top_k: int = Query(default=5),
) -> Dict[str, object]:
    if not filename.lower().endswith(".wav"):
        raise HTTPException(status_code=400, detail="Only WAV uploads are supported")

    try:
        audio = _decode_wav_bytes(audio_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except wave.Error as exc:
        raise HTTPException(status_code=400, detail="Invalid WAV payload") from exc

    query_fp = generate_fingerprint(audio)

    with _index_lock:
        matches = match_fingerprint(query_fp, _fingerprint_db)

    return {
        "results": [
            {"track_id": track_id, "score": score, "metadata": _track_store.get_track(track_id)}
            for track_id, score in matches[: max(0, int(top_k))]
        ]
    }


@app.post("/ingest/dataset")
def ingest_dataset(path: str = Query(...)) -> Dict[str, object]:
    with _index_lock:
        queued, failed = enqueue_all_audio_files(
            path=path,
            queue=_job_queue,
            embedding_path=_VECTOR_INDEX_PATH,
            progress_tracker=_ingestion_tracker,
        )
        status = _ingestion_tracker.snapshot(track_store=_track_store)

    return {
        "queued_files": int(queued),
        "failed_files": int(failed),
        "status": status,
    }


@app.get("/ingest/status")
def ingest_status() -> Dict[str, int]:
    with _index_lock:
        return _ingestion_tracker.snapshot(track_store=_track_store)


@app.post("/search")
def search(payload: SearchRequest) -> Dict[str, object]:
    started = time.perf_counter()
    try:
        results = semantic_search(
            query_audio_path=payload.query_audio_path,
            top_k=payload.top_k,
            vector_index_path=_VECTOR_INDEX_PATH,
            track_store=_track_store,
            filters=payload.filters,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    metrics_registry.search_latency = float(time.perf_counter() - started)
    cache = search_cache_metrics()
    metrics_registry.update_cache_hit_rate(cache["hits"], cache["misses"])
    return {"results": results}


@app.post("/search/batch")
def search_batch(payload: BatchSearchRequest) -> Dict[str, object]:
    started = time.perf_counter()
    results = semantic_search_batch(
        queries=payload.queries,
        top_k=payload.top_k,
        vector_index_path=_VECTOR_INDEX_PATH,
        track_store=_track_store,
    )
    metrics_registry.search_latency = float(time.perf_counter() - started)
    cache = search_cache_metrics()
    metrics_registry.update_cache_hit_rate(cache["hits"], cache["misses"])
    return {"results": results}


@app.post("/search/text")
def search_text(payload: TextSearchRequest) -> Dict[str, object]:
    results = search_by_text(
        query_text=payload.query,
        top_k=payload.top_k,
        vector_index_path=_VECTOR_INDEX_PATH,
        track_store=_track_store,
    )
    return {"results": results}


@app.post("/search/lyrics")
def search_lyrics(payload: TextSearchRequest) -> Dict[str, object]:
    results = search_by_lyrics(
        lyrics=payload.query,
        top_k=payload.top_k,
        vector_index_path=_VECTOR_INDEX_PATH,
        track_store=_track_store,
    )
    return {"results": results}


@app.get("/search/cache/metrics")
def cache_metrics() -> Dict[str, int]:
    return search_cache_metrics()


@app.get("/metrics")
def metrics_endpoint() -> PlainTextResponse:
    metrics_registry.queue_depth = _queue_depth()
    cache = search_cache_metrics()
    metrics_registry.update_cache_hit_rate(cache["hits"], cache["misses"])
    return PlainTextResponse(metrics_registry.prometheus(), media_type="text/plain; version=0.0.4")


@app.get("/web")
def web_index() -> FileResponse:
    index_path = _WEB_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Web UI not found")
    return FileResponse(str(index_path))


@app.post("/dataset/register")
def dataset_register(payload: DatasetRegisterRequest) -> Dict[str, object]:
    dataset = _dataset_manager.register_dataset(payload.dataset_id, payload.name)
    return {"dataset": dataset}


@app.get("/experiments")
def experiments() -> Dict[str, object]:
    return {"experiments": _experiment_router.list_experiments()}


@app.post("/experiments/create")
def experiments_create(payload: ExperimentCreateRequest) -> Dict[str, object]:
    experiment = _experiment_router.create_experiment(
        experiment_id=payload.experiment_id,
        name=payload.name,
        model_a=payload.model_a,
        model_b=payload.model_b,
    )
    return {"experiment": experiment}


@app.post("/dataset/ingest")
def dataset_ingest(payload: DatasetIngestRequest) -> Dict[str, object]:
    _dataset_manager.set_status(payload.dataset_id, "ingesting")
    with _index_lock:
        queued, failed = enqueue_all_audio_files(
            path=payload.path,
            queue=_job_queue,
            embedding_path=_VECTOR_INDEX_PATH,
            progress_tracker=_ingestion_tracker,
        )
    _dataset_manager.set_status(payload.dataset_id, "queued", track_count=queued)
    metrics_registry.start_ingestion_timer()
    metrics_registry.record_ingested(queued)
    return {"dataset_id": payload.dataset_id, "queued_files": queued, "failed_files": failed}


@app.get("/dataset/list")
def dataset_list() -> Dict[str, object]:
    return {"datasets": _dataset_manager.list_datasets()}


@app.get("/dataset/status")
def dataset_status(dataset_id: str | None = Query(default=None)) -> Dict[str, object]:
    if dataset_id:
        return {"dataset": _dataset_manager.get_dataset(dataset_id)}
    return {"datasets": _dataset_manager.list_datasets()}


@app.post("/recognize")
def recognize(payload: RecognizeRequest) -> Dict[str, object]:
    try:
        query_fp = generate_fingerprints(payload.audio_path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    with _index_lock:
        _sync_fingerprint_index()
        matches = _fingerprint_index.query(query_fp)

    if not matches:
        return {"track_id": None, "score": 0.0, "offset": 0.0}

    best = matches[0]
    return {
        "track_id": best["track_id"],
        "score": float(best["score"]),
        "offset": float(best["offset"]),
        "metadata": _track_store.get_track(str(best["track_id"])),
    }


@app.post("/recognize/stream")
def recognize_stream(payload: StreamRecognizeRequest) -> Dict[str, object]:
    with _index_lock:
        _sync_fingerprint_index()
        _stream_recognizer.sample_rate = int(payload.sample_rate)
        matches = _stream_recognizer.push_chunk(payload.samples)
    if not matches:
        return {"track_id": None, "score": 0.0, "offset": 0.0}
    best = matches[0]
    return {
        "track_id": best["track_id"],
        "score": float(best["score"]),
        "offset": float(best["offset"]),
        "metadata": _track_store.get_track(str(best["track_id"])),
    }


@app.post("/index/add_track")
def index_add_track(payload: AddTrackRequest) -> Dict[str, object]:
    try:
        with _index_lock:
            record = _index_updater.add_track(audio_path=payload.audio_path, metadata=payload.metadata)
            _recommendation_engine.build_similarity_graph()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"track": record}


@app.get("/recommend/{track_id}")
def recommend(track_id: str, k: int = Query(default=10)) -> Dict[str, object]:
    with _index_lock:
        recommendations = _recommendation_engine.recommend(track_id=track_id, k=k)
    return {
        "track_id": track_id,
        "results": [
            {"track_id": rec_id, "score": score, "metadata": _track_store.get_track(rec_id)}
            for rec_id, score in recommendations
        ],
    }


@app.get("/playlist/generate")
def playlist_generate(seed_track_id: str, length: int = Query(default=10)) -> Dict[str, object]:
    with _index_lock:
        playlist = generate_playlist(
            seed_track_id=seed_track_id,
            length=length,
            recommendation_engine=_recommendation_engine,
            track_store=_track_store,
        )
    return {"seed_track_id": seed_track_id, "playlist": playlist}


@app.post("/assistant/query")
def assistant_query(payload: AssistantQueryRequest) -> Dict[str, object]:
    parsed = handle_query(payload.prompt)
    return {"response": parsed}


@app.post("/events")
def events(payload: EventRequest) -> Dict[str, object]:
    # Process synchronously for deterministic API behavior; stream remains available for async consumers.
    _event_handler(dict(payload.event))
    return {"status": "queued"}


@app.get("/recommendations/home")
def recommendations_home(user_id: str, k: int = Query(default=10)) -> Dict[str, object]:
    user_features = _feature_store.get_user_features(user_id) or {}
    target_k = max(1, int(k))
    gather_k = max(10, target_k * 5)
    _rebuild_graph_from_interactions()
    session_history = _session_model.sessions.get(str(user_id), [])
    user_vec = user_features.get("embedding")
    candidate_scores = _realtime_recommender.generate_candidates(
        user_id=user_id,
        user_embedding=user_vec if isinstance(user_vec, list) else None,
        session_history=list(session_history),
        top_k=gather_k,
    )

    candidates = []
    for track_id, base_score in candidate_scores.items():
        feats = _feature_store.get_track_features(track_id) or {}
        if "embedding" not in feats and track_id in _vector_index_service.track_vectors:
            feats["embedding"] = _vector_index_service.track_vectors[track_id].tolist()
        item = dict(feats)
        item["track_id"] = track_id
        item["score"] = float(base_score)
        item["metadata"] = _track_store.get_track(track_id)
        candidates.append(item)

    ranked = rank_tracks(user_features, candidates)
    _refresh_user_clusters()
    cluster_id = _taste_clusterer.assign_user_cluster(user_id)
    cluster_pref_tracks = set(_taste_clusterer.recommend_from_cluster(cluster_id, k=max(50, gather_k)))
    q_values = _rl_recommender.q_values.get(str(user_id), {})
    for item in ranked:
        tid = str(item.get("track_id"))
        rl_bonus = float(q_values.get(tid, 0.0))
        cluster_boost = 0.1 if tid in cluster_pref_tracks else 0.0
        rank_score = float(item.get("rank_score", 0.0))
        base_score = float(item.get("score", 0.0))
        item["score"] = 0.8 * rank_score + 0.2 * base_score + rl_bonus + cluster_boost
        item["cluster_boost"] = cluster_boost
        item["cluster_id"] = cluster_id
    ranked.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
    return {"user_id": user_id, "results": ranked[:target_k]}


@app.get("/recommendations/session")
def recommendations_session(user_id: str) -> Dict[str, object]:
    recs = _session_model.recommend(user_id)
    return {
        "user_id": user_id,
        "results": [{"track_id": tid, "metadata": _track_store.get_track(tid)} for tid in recs],
    }


@app.get("/recommendations/realtime")
def recommendations_realtime(user_id: str, k: int = Query(default=10)) -> Dict[str, object]:
    user_features = _feature_store.get_user_features(user_id) or {}
    user_vec = user_features.get("embedding")
    session_history = _session_model.sessions.get(str(user_id), [])
    _rebuild_graph_from_interactions()
    candidates = _realtime_recommender.top_candidates(
        user_id=user_id,
        user_embedding=user_vec if isinstance(user_vec, list) else None,
        session_history=list(session_history),
        top_k=k,
    )
    return {
        "user_id": user_id,
        "results": [
            {"track_id": track_id, "score": score, "metadata": _track_store.get_track(track_id)}
            for track_id, score in candidates
        ],
    }


@app.get("/recommendations/gnn")
def recommendations_gnn(user_id: str, k: int = Query(default=10)) -> Dict[str, object]:
    _rebuild_graph_from_interactions()
    results = _gnn_recommender.recommend_with_gnn(user_id=user_id, k=k)
    return {
        "user_id": user_id,
        "results": [
            {"track_id": track_id, "score": score, "metadata": _track_store.get_track(track_id)}
            for track_id, score in results
        ],
    }


@app.get("/recommendations/discovery")
def recommendations_discovery(user_id: str, k: int = Query(default=10)) -> Dict[str, object]:
    user_features = _feature_store.get_user_features(user_id) or {}
    target_k = max(1, int(k))
    gather_k = max(20, target_k * 6)
    _rebuild_graph_from_interactions()

    session_history = _session_model.sessions.get(str(user_id), [])
    user_vec = user_features.get("embedding")
    candidate_scores = _realtime_recommender.generate_candidates(
        user_id=user_id,
        user_embedding=user_vec if isinstance(user_vec, list) else None,
        session_history=list(session_history),
        top_k=gather_k,
    )
    for track_id, score in _gnn_recommender.recommend_with_gnn(user_id=user_id, k=gather_k):
        candidate_scores[str(track_id)] = max(candidate_scores.get(str(track_id), 0.0), float(score))

    candidates = []
    seed_track = str(session_history[-1]) if session_history else ""
    seed_genome = _music_genome_store.get_genome(seed_track) if seed_track else None
    user_foundation_vec = _intelligence_foundation_model.embed_features(user_vec if isinstance(user_vec, list) else [0.0])
    for track_id, base_score in candidate_scores.items():
        feats = _feature_store.get_track_features(track_id) or {}
        if "embedding" not in feats and track_id in _vector_index_service.track_vectors:
            feats["embedding"] = _vector_index_service.track_vectors[track_id].tolist()
        row = dict(feats)
        row["track_id"] = track_id
        row["score"] = float(base_score)
        row["trending_score"] = float(base_score)
        row["metadata"] = _track_store.get_track(track_id)
        rt = _realtime_feature_aggregator.compute_features(track_id)
        row["engagement_score"] = float(rt.get("engagement_score", 0.0))
        candidate_genome = _music_genome_store.get_genome(track_id)
        row["genome_similarity"] = (
            float(_music_genome_engine.similarity_score(seed_genome, candidate_genome))
            if seed_genome and candidate_genome
            else 0.0
        )
        candidate_foundation = _intelligence_foundation_model.embed_features(
            [
                float(hash(str((row.get("metadata") or {}).get("genre", ""))) % 100) / 100.0,
                float(hash(str((row.get("metadata") or {}).get("artist", ""))) % 100) / 100.0,
            ]
        )
        row["foundation_embedding_similarity"] = float(np.dot(user_foundation_vec, candidate_foundation))
        candidates.append(row)

    ranked = rank_tracks(user_features, candidates)
    q_values = _rl_recommender.q_values.get(str(user_id), {})
    for item in ranked:
        tid = str(item.get("track_id"))
        item["rl_score"] = float(q_values.get(tid, 0.0))

    ranked = _bandit_recommender.score_candidates(user_id=user_id, candidates=ranked)
    _refresh_user_clusters()
    cluster_id = _taste_clusterer.assign_user_cluster(user_id)
    cluster_pref_tracks = set(_taste_clusterer.recommend_from_cluster(cluster_id, k=max(50, gather_k)))
    final = _discovery_feed_ranker.rank(user_id=user_id, candidates=ranked, cluster_pref_tracks=cluster_pref_tracks)
    selected_engine = _streaming_reco_orchestrator.choose_best_engine({"cold_start": len(session_history) == 0})
    return {"user_id": user_id, "cluster_id": cluster_id, "engine": selected_engine, "results": final[:target_k]}


@app.get("/recommendations/graph")
def recommendations_graph(user_id: str, k: int = Query(default=10)) -> Dict[str, object]:
    _rebuild_graph_from_interactions()
    results = _graph_recommender.recommend_from_graph(user_id, k)
    return {
        "user_id": user_id,
        "results": [
            {"track_id": track_id, "score": score, "metadata": _track_store.get_track(track_id)}
            for track_id, score in results
        ],
    }


@app.get("/tracks/trending")
def tracks_trending(top_k: int = Query(default=20)) -> Dict[str, object]:
    results = _trend_detector.detect_trending_tracks(top_k=top_k)
    return {
        "results": [
            {
                "track_id": item["track_id"],
                "count": item["count"],
                "metadata": _track_store.get_track(str(item["track_id"])),
            }
            for item in results
        ]
    }


@app.get("/tracks/viral")
def tracks_viral(window_hours: int = Query(default=24), top_k: int = Query(default=20)) -> Dict[str, object]:
    viral = _trend_detector.detect_viral_tracks(window_hours=window_hours, top_k=max(50, top_k))
    scored = []
    for item in viral:
        track_id = str(item.get("track_id"))
        if not track_id:
            continue
        features = _feature_store.get_track_features(track_id) or {}
        features["velocity"] = float(item.get("velocity", 0.0))
        score = _viral_predictor.predict_track_virality(features)
        scored.append(
            {
                "track_id": track_id,
                "score": float(score),
                "velocity": float(item.get("velocity", 0.0)),
                "metadata": _track_store.get_track(track_id),
            }
        )
    scored.sort(key=lambda x: x["score"], reverse=True)
    return {"results": scored[: max(0, int(top_k))]}


@app.get("/users/cluster")
def users_cluster(user_id: str) -> Dict[str, object]:
    clusters = _refresh_user_clusters()
    cluster_id = _taste_clusterer.assign_user_cluster(user_id)
    return {
        "user_id": user_id,
        "cluster_id": cluster_id,
        "cluster_size": len([uid for uid, cid in clusters.items() if cid == cluster_id]),
        "cluster_recommendations": _taste_clusterer.recommend_from_cluster(cluster_id, k=10),
    }


@app.get("/artists/similar")
def artists_similar(artist_id: str | None = Query(default=None), user_id: str | None = Query(default=None), k: int = Query(default=10)) -> Dict[str, object]:
    _artist_intelligence.build_artist_graph(_track_store.list_tracks())
    if artist_id:
        results = _artist_intelligence.compute_artist_similarity(artist_id, top_k=k)
        return {"artist_id": artist_id, "results": [{"artist_id": aid, "score": score} for aid, score in results]}
    if user_id:
        results = _artist_intelligence.recommend_artists(user_id=user_id, k=k)
        return {"user_id": user_id, "results": [{"artist_id": aid, "score": score} for aid, score in results]}
    raise HTTPException(status_code=400, detail="artist_id or user_id is required")


@app.get("/artists/intelligence")
def artists_intelligence(user_id: str | None = Query(default=None), artist_id: str | None = Query(default=None), k: int = Query(default=10)) -> Dict[str, object]:
    graph = _artist_intelligence.build_artist_graph(_track_store.list_tracks())
    if artist_id:
        similar = _artist_intelligence.compute_artist_similarity(artist_id=artist_id, top_k=k)
        return {"artist_id": artist_id, "graph_nodes": len(graph), "similar": [{"artist_id": a, "score": s} for a, s in similar]}
    if user_id:
        recs = _artist_intelligence.recommend_artists(user_id=user_id, k=k)
        return {"user_id": user_id, "graph_nodes": len(graph), "recommended_artists": [{"artist_id": a, "score": s} for a, s in recs]}
    return {"graph_nodes": len(graph)}


@app.post("/features/user")
def features_user(payload: UserFeatureRequest) -> Dict[str, object]:
    _feature_store.store_user_features(payload.user_id, payload.features)
    return {"user_id": payload.user_id, "features": _feature_store.get_user_features(payload.user_id)}


@app.post("/features/track")
def features_track(payload: TrackFeatureRequest) -> Dict[str, object]:
    _feature_store.store_track_features(payload.track_id, payload.features)
    return {"track_id": payload.track_id, "features": _feature_store.get_track_features(payload.track_id)}


@app.post("/index/add")
def index_add(payload: IndexAddRequest) -> Dict[str, object]:
    _vector_index_service.add_vector(payload.track_id, payload.vector)
    _vector_index_service.save_index()
    return {"track_id": payload.track_id, "dimension": len(payload.vector)}


@app.post("/index/search")
def index_search(payload: IndexSearchRequest) -> Dict[str, object]:
    results = _vector_index_service.search(payload.vector, payload.k)
    return {"results": [{"track_id": tid, "score": score} for tid, score in results]}


@app.post("/rl/reward")
def rl_reward(payload: RLRewardRequest) -> Dict[str, object]:
    _rl_recommender.update_reward(payload.user_id, payload.track_id, payload.reward)
    return {
        "user_id": payload.user_id,
        "track_id": payload.track_id,
        "reward": float(payload.reward),
    }


@app.post("/bandit/reward")
def bandit_reward(payload: BanditRewardRequest) -> Dict[str, object]:
    _bandit_recommender.update_reward(payload.user_id, payload.track_id, payload.reward)
    return {"user_id": payload.user_id, "track_id": payload.track_id, "reward": float(payload.reward)}


@app.post("/embeddings/train")
def embeddings_train(payload: EmbeddingTrainRequest) -> Dict[str, object]:
    interactions = _online_learning.list_interactions()
    prepared = []
    for item in interactions:
        event_type = str(item.get("event") or "")
        reward = 1.0 if event_type == "play" else (1.5 if event_type == "click" else 2.0)
        prepared.append({"track_id": item.get("track_id"), "reward": reward})

    if payload.incremental:
        embeddings = _embedding_trainer.update_embeddings_incrementally(prepared)
    else:
        embeddings = _embedding_trainer.train_track_embeddings(prepared)
    path = _embedding_trainer.save_embeddings() if payload.save else _embedding_trainer.output_path
    return {"tracks": len(embeddings), "path": path, "incremental": bool(payload.incremental)}


@app.post("/embeddings/batch-train")
def embeddings_batch_train(payload: BatchEmbeddingTrainRequest) -> Dict[str, object]:
    interactions = _online_learning.list_interactions()
    prepared = []
    for item in interactions:
        event_type = str(item.get("event") or "")
        reward = 1.0 if event_type == "play" else (1.5 if event_type == "click" else 2.0)
        prepared.append({"track_id": item.get("track_id"), "reward": reward})
    trained = _distributed_embedding_pipeline.batch_train_embeddings(prepared)
    shards = _distributed_embedding_pipeline.shard_embeddings(num_shards=payload.shards)
    if not payload.save:
        trained["path"] = _embedding_trainer.output_path
    return {"training": trained, "shards": {str(k): len(v) for k, v in shards.items()}}


@app.get("/ai-dj/session")
def ai_dj_session(user_id: str, persona: str = Query(default="classic"), k: int = Query(default=10)) -> Dict[str, object]:
    discovery = recommendations_discovery(user_id=user_id, k=k)
    recommendations = list(discovery.get("results", []))
    return _ai_dj_service.create_session(user_id=user_id, recommendations=recommendations, persona=persona)


@app.get("/ai-dj/radio")
def ai_dj_radio(user_id: str, context: str = Query(default="focus"), k: int = Query(default=12)) -> Dict[str, object]:
    discovery = recommendations_discovery(user_id=user_id, k=k)
    persona = "energy" if context in {"workout", "party"} else ("chill" if context in {"focus", "study"} else "classic")
    session = _ai_dj_service.create_session(user_id=user_id, recommendations=list(discovery.get("results", [])), persona=persona)
    session["context"] = context
    return session


@app.post("/music/generate")
def music_generate(payload: MusicGenerateRequest) -> Dict[str, object]:
    return _generation_service.generate(prompt=payload.prompt, seconds=payload.seconds)


@app.post("/music/style-transfer")
def music_style_transfer(payload: StyleTransferRequest) -> Dict[str, object]:
    return _generation_service.style_transfer(track=payload.track, style=payload.style)


@app.post("/music/remix")
def music_remix(payload: RemixRequest) -> Dict[str, object]:
    return _generation_orchestrator.remix(track=payload.track, style=payload.style)


@app.post("/music/generate-advanced")
def music_generate_advanced(payload: AdvancedGenerateRequest) -> Dict[str, object]:
    return _generation_orchestrator.generate_advanced(prompt=payload.prompt, mode=payload.mode, seconds=payload.seconds)


@app.get("/system/streaming-status")
def streaming_status() -> Dict[str, object]:
    return {
        "inference": _streaming_inference.status(),
        "service": _streaming_service.status(),
        "feature_tracks": len(_realtime_feature_aggregator.snapshot()),
    }


@app.get("/ai/taste-profile")
def ai_taste_profile(user_id: str) -> Dict[str, object]:
    return {
        "user_id": user_id,
        "summary": _taste_llm.generate_taste_summary(user_id),
        "predictions": _taste_llm.predict_next_genres_artists(user_id),
    }


@app.get("/ai/taste-explanation")
def ai_taste_explanation(user_id: str, k: int = Query(default=5)) -> Dict[str, object]:
    recs = recommendations_discovery(user_id=user_id, k=k).get("results", [])
    explanation = _taste_llm.explain_recommendations(user_id=user_id, tracks=recs)
    return {"user_id": user_id, "explanation": explanation}


@app.get("/trends/global")
def trends_global(top_k: int = Query(default=20)) -> Dict[str, object]:
    return {
        "trending_tracks": _global_trend_intel.detect_global_trends(top_k=top_k),
        "clusters": _global_trend_intel.cluster_trending_tracks(top_k=max(20, top_k)),
        "emerging_genres": _global_trend_intel.identify_emerging_genres(top_k=10),
    }


@app.get("/artists/breakout")
def artists_breakout(top_k: int = Query(default=10)) -> Dict[str, object]:
    breakout = _artist_growth_predictor.predict_breakout_artists(top_k=top_k)
    _ai_label_engine.discover_new_artists(breakout, top_k=top_k)
    return {"results": breakout}


@app.get("/music/knowledge-graph")
def music_knowledge_graph() -> Dict[str, object]:
    return _knowledge_graph_builder.graph()


@app.get("/agents/status")
def agents_status() -> Dict[str, object]:
    discovery = recommendations_discovery(user_id="system", k=5).get("results", [])
    trends = _global_trend_intel.detect_global_trends(top_k=5)
    _music_agent_system.run_discovery_agent(discovery)
    _music_agent_system.run_playlist_agent(discovery, length=5)
    _music_agent_system.run_trend_agent(trends)
    _music_agent_system.run_generation_agent(["chill synth", "orchestral pulse"])
    return _music_agent_system.status()


@app.get("/system/autonomous-optimization")
def system_autonomous_optimization() -> Dict[str, object]:
    metrics = {
        "ctr": float(metrics_registry.cache_hit_rate),
        "latency": float(metrics_registry.search_latency or 0.0),
    }
    _autonomous_optimizer.monitor_metrics(metrics)
    update = _autonomous_optimizer.optimize()
    _bandit_recommender.epsilon = float(update["exploration"])
    return {"metrics": metrics, "optimization": update, "orchestrator": _streaming_reco_orchestrator.status()}


@app.get("/system/recommendation-stats")
def recommendation_stats() -> Dict[str, object]:
    interactions = _online_learning.list_interactions()
    clusters = _refresh_user_clusters()
    return {
        "interactions": len(interactions),
        "users_with_clusters": len(clusters),
        "cluster_count": len(set(clusters.values())) if clusters else 0,
        "tracked_embeddings": len(_embedding_trainer.track_embeddings),
        "trending_tracks": len(_trend_detector.detect_trending_tracks(top_k=100)),
        "viral_tracks": len(_trend_detector.detect_viral_tracks(window_hours=24, top_k=100)),
    }


@app.get("/ai/system-status")
def ai_system_status() -> Dict[str, object]:
    return {
        "self_evolving_recommender": _self_evolving_recommender.metrics_snapshot(),
        "meta_learning_engine": _meta_learning_engine.metrics_snapshot(),
        "autonomous_experimentation": _autonomous_experimentation.metrics_snapshot(),
        "auto_scaling_ai": _auto_scaling_ai.metrics_snapshot(),
        "distributed_training": _distributed_training_orchestrator.status(),
        "model_evolution": _model_evolution_manager.metrics_snapshot(),
    }


@app.get("/ai/self-evolution")
def ai_self_evolution() -> Dict[str, object]:
    supervisor_status = _ai_music_supervisor.run_supervision_cycle(
        performance_score=float(metrics_registry.cache_hit_rate or 0.6)
    )
    loop_status = _self_improvement_loop.run_iteration(performance_score=float(metrics_registry.cache_hit_rate or 0.6))
    return {"supervisor": supervisor_status, "loop": loop_status}


@app.get("/ai/simulation/run")
def ai_simulation_run() -> Dict[str, object]:
    eco = _ecosystem_simulator.simulate_users_artists_playlists()
    trends = _ecosystem_simulator.simulate_trend_emergence()
    return {"ecosystem": eco, "trends": trends, "metrics": _ecosystem_simulator.metrics_snapshot()}


@app.get("/ai/virtual-artists")
def ai_virtual_artists() -> Dict[str, object]:
    return {"artists": _virtual_artist_factory.list_artists(), "metrics": _virtual_artist_factory.metrics_snapshot()}


@app.post("/ai/artist/create")
def ai_artist_create(payload: CreateArtistRequest) -> Dict[str, object]:
    factory = VirtualArtistFactory()
    try:
        artist = factory.create_artist(name=payload.name, genre=payload.genre, persona=payload.persona)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    _ai_music_supervisor.log_event("artist_created", artist)
    return {"status": "created", "artist": artist}


@app.post("/ai/music/produce")
def ai_music_produce(payload: ProduceMusicRequest) -> Dict[str, object]:
    engine = AIProducerEngine()
    try:
        track = engine.produce_track(genre=payload.genre, mood=payload.mood, tempo=payload.tempo, genome=payload.genome)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    _ai_music_supervisor.log_event("music_generated", track)
    return {"status": "generated", "track": track}


@app.post("/ai/music/collaborate")
def ai_music_collaborate(payload: AIMusicCollaborateRequest) -> Dict[str, object]:
    return _collaborative_ai_band.create_track_collaboratively(prompt=payload.prompt, rounds=payload.rounds)


@app.get("/ai/culture/trends")
def ai_culture_trends() -> Dict[str, object]:
    interactions = _online_learning.list_interactions()
    events = []
    for item in interactions:
        events.append({"genre": str(((_track_store.get_track(str(item.get("track_id") or "")) or {}).get("genre") or "unknown"))})
    shifts = _culture_trend_engine.detect_cultural_music_shifts(events=events, top_k=10)
    evolution = _culture_trend_engine.track_genre_evolution()
    return {"shifts": shifts, "evolution": evolution}


@app.get("/ai/music/map")
def ai_music_map() -> Dict[str, object]:
    interactions = _online_learning.list_interactions()
    events = []
    for idx, item in enumerate(interactions):
        track = _track_store.get_track(str(item.get("track_id") or "")) or {}
        region = "global" if idx % 2 == 0 else "regional"
        events.append({"region": region, "genre": str(track.get("genre") or "unknown")})
    mapped = _global_music_map.map_regional_music_taste(events)
    return {"map": mapped, "top_genres": _global_music_map.top_genre_by_region()}


@app.get("/ai/experiments/autonomous")
def ai_experiments_autonomous() -> Dict[str, object]:
    launched = _autonomous_experimentation.launch_ab_experiment("rank_v1", "rank_v2", split=0.5)
    evaluated = _autonomous_experimentation.evaluate_experiment(
        launched["experiment_id"],
        scores_a=[0.6, 0.65, 0.7],
        scores_b=[0.62, 0.64, 0.63],
    )
    deployed = _autonomous_experimentation.deploy_best_performer(launched["experiment_id"])
    _meta_learning_engine.store_experiment_results(
        pipeline_name=str(deployed or "rank_v1"),
        metrics={"quality": max(evaluated["mean_a"], evaluated["mean_b"]), "engagement": 0.6, "latency": 0.2},
        metadata={"experiment_id": launched["experiment_id"]},
    )
    return {
        "launched": launched,
        "evaluated": evaluated,
        "deployed": deployed,
        "meta_recommendation": _meta_learning_engine.recommend_best_model(),
    }


@app.get("/ai/supervisor/status")
def ai_supervisor_status() -> Dict[str, object]:
    return {
        "supervisor": _ai_music_supervisor.status(),
        "self_improvement_loop": _self_improvement_loop.collect_metrics(),
        "market_snapshot": _market_simulator.simulate_streaming_platform_economy(streams=100000),
    }


@app.post("/ai/music/analyze")
def ai_music_analyze(payload: AnalyzeMusicRequest) -> Dict[str, object]:
    if payload.audio_path:
        embedding = _intelligence_foundation_model.embed_audio(payload.audio_path)
        structure = _intelligence_foundation_model.analyze_structure(payload.audio_path)
        genome = _music_genome_engine.extract_genome(payload.audio_path)
    elif payload.midi_path:
        embedding = _intelligence_foundation_model.embed_midi(payload.midi_path)
        structure = _intelligence_foundation_model.analyze_structure(payload.midi_path)
        genome = _music_genome_engine.extract_genome(payload.midi_path)
    elif payload.feature_vector is not None:
        embedding = _intelligence_foundation_model.embed_features(payload.feature_vector)
        structure = _intelligence_foundation_model.analyze_structure("features://vector")
        genome = _music_genome_engine.extract_genome("features://vector")
    else:
        raise HTTPException(status_code=400, detail="audio_path, midi_path, or feature_vector is required")
    return {
        "foundation_embedding_summary": {
            "dimension": int(embedding.shape[0]),
            "norm": float(np.linalg.norm(embedding)),
            "backend": _intelligence_foundation_model.backend,
        },
        "structure": structure,
        "music_genome": genome,
    }


@app.post("/ai/music/genome")
def ai_music_genome(payload: MusicGenomeRequest) -> Dict[str, object]:
    genome = _music_genome_engine.extract_genome(payload.audio_path)
    return {"music_genome": genome}


@app.get("/ai/music/similar/{track_id}")
def ai_music_similar(track_id: str, top_k: int = Query(default=10)) -> Dict[str, object]:
    genome = _music_genome_store.get_genome(track_id)
    if not genome:
        raise HTTPException(status_code=404, detail="Track genome not found")
    results = _music_genome_store.search_similar(genome, top_k=top_k)
    return {"track_id": track_id, "results": results}


@app.post("/ai/music/generate-from-genome")
def ai_music_generate_from_genome(payload: GenerateFromGenomeRequest) -> Dict[str, object]:
    generated = _ai_producer_engine.generate_track_from_genome(payload.genome)
    return generated


@app.get("/ai/intelligence/trends")
def ai_intelligence_trends(top_k: int = Query(default=3)) -> Dict[str, object]:
    report = _global_music_intelligence.generate_trend_report(top_k=top_k)
    return report


@app.post("/ai/intelligence/predict")
def ai_intelligence_predict(payload: IntelligencePredictRequest) -> Dict[str, object]:
    emotional_intensity = payload.emotional_intensity
    if emotional_intensity is None:
        emotional_intensity = float(
            np.clip((payload.energy + payload.mood + payload.harmonic_density) / 3.0, 0.0, 1.0)
        )
    vector = MusicFeatureVector(
        tempo=payload.tempo,
        energy=payload.energy,
        mood=payload.mood,
        novelty=payload.novelty,
        rhythm_complexity=payload.rhythm_complexity,
        harmonic_density=payload.harmonic_density,
        emotional_intensity=emotional_intensity,
    )
    try:
        hit_probability = _global_music_intelligence.predict_hit_probability(vector)
        expected_virality = _global_music_intelligence.estimate_expected_virality(vector)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    prediction = HitPrediction(
        track_id=payload.track_id or f"track_pred_{int(payload.tempo * 10)}_{int(payload.energy * 100)}",
        hit_probability=hit_probability,
        expected_virality=expected_virality,
    )
    return prediction.__dict__


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
