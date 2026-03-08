# Mantra Architecture

Mantra is a structural music similarity engine designed to analyze melodic patterns inside MIDI data.

Unlike audio fingerprinting systems, Mantra focuses on **melodic interval structures**.

---

# System Overview

Mantra pipeline:

MIDI File  
↓  
Pitch Extraction  
↓  
Interval Normalization  
↓  
Fingerprint Generation  
↓  
Similarity Search  
↓  
Similarity Score

---

# Core Modules

## extraction

Responsible for reading MIDI files and extracting note pitch sequences.

Key tasks:

- parse MIDI events
- extract note pitches
- preserve melodic order

---

## core_algorithms

Contains the fundamental music analysis algorithms.

Main algorithm:

Interval normalization.

Example:

Melody:

60 62 64 65

Intervals:

+2 +2 +1

This makes the melody **key invariant**.

---

## fingerprint

Transforms interval sequences into compact fingerprint vectors.

Purpose:

- efficient comparison
- hashing
- indexing

---

## indexing

Implements fast search structures.

Used for:

- storing fingerprints
- retrieving candidate matches

Possible future algorithms:

- LSH (Locality Sensitive Hashing)
- ANN search
- vector similarity

---

## matching

Responsible for computing similarity scores.

Methods include:

- interval overlap
- structural similarity
- regression testing

---

## persistence

Handles storage of fingerprints.

Current implementation:

SQLite database.

Future:

- distributed database
- vector database
- cloud storage

---

## app

FastAPI service exposing Mantra functionality through HTTP API.

Endpoints allow:

- MIDI upload
- similarity search
- fingerprint inspection

---

# Data Flow

Example workflow:

1. User uploads MIDI file
2. System extracts note pitches
3. Intervals are computed
4. Fingerprint is generated
5. Fingerprint compared against database
6. Similar tracks returned

---

# Why Interval Fingerprinting

Interval representation provides:

Key invariance  
Tempo independence  
Structural similarity detection  

Example:

C major melody

C D E G

Transposed melody

F G A C

Both produce identical interval structures.

---

# Future Architecture

Next stages of Mantra development:

Phase 1  
MIDI structural similarity

Phase 2  
Audio fingerprinting

Phase 3  
Large-scale similarity search

Phase 4  
Cloud API

Phase 5  
DAW plugin (VST3)

---

# Long-Term Vision

Mantra aims to become a professional tool for:

music producers  
record labels  
music publishers  
copyright analysis  

The system may evolve into a hybrid platform combining:

- structural music analysis
- AI similarity detection
- audio fingerprinting
---

# Phase 1 Architecture Expansion

## dataset_engine

Path: `mantra/dataset_engine/`

Responsibilities:

- scan MIDI datasets (`scan_dataset`)
- load MIDI files (`load_midi`)
- extract melody notes (`extract_melody`)
- derive pitch sequences (`extract_pitch_sequence`)
- derive normalized interval sequences (`extract_interval_sequence`)

This layer is independently testable and reuses the existing interval normalization pipeline in `core_algorithms.interval`.

---

## explain_engine

Path: `mantra/explain_engine/`

Main function: `explain_similarity(trackA, trackB)`

Explanation signals:

- interval overlap
- shared interval n-gram patterns
- pitch similarity (reusing existing pitch similarity logic)
- rhythm similarity (duration-based heuristic)

Returns a structured payload with numeric metrics for explainability and debugging.

---

## CLI Interface

Path: `mantra/interfaces/cli/mantra_cli.py`

Commands:

- `mantra index <folder>`
- `mantra search <midi>`
- `mantra explain <midiA> <midiB>`

The CLI orchestrates existing fingerprinting + similarity engine modules and is covered by dedicated pytest tests.

# Phase 2 Foundation (Audio + Vector)

## audio_engine

Path: `mantra/audio_engine/`

Responsibilities:

- load WAV audio files (`load_audio`)
- extract chroma summaries (`extract_chroma`)
- estimate global tempo (`extract_tempo`)
- derive fixed-length pitch contour (`extract_pitch_contour`)
- compute spectral statistics (`extract_spectral_features`)

This module provides the first audio-analysis building blocks while keeping the existing MIDI pipeline unchanged.

---

## vector_engine

Path: `mantra/vector_engine/`

Responsibilities:

- assemble extracted audio features into one fixed-length numeric embedding (`build_music_embedding`)
- guarantee stable embedding size for indexing/search infrastructure

Current embedding layout:

- chroma summary (12)
- tempo (1)
- pitch contour (64)
- spectral statistics (10)

Total: 87 float32 values.

---

# Phase 3 Foundation (Vector Similarity Index)

## vector_index

Path: `mantra/vector_index/`

Responsibilities:

- maintain an embedding index for high-speed nearest-neighbor search (`VectorIndex`)
- add track embeddings (`add`)
- run top-k similarity retrieval (`search`)
- persist and restore indexes (`save`, `load`)

Backend strategy:

- FAISS (`IndexFlatIP`) when available for accelerated vector search
- numpy cosine-similarity fallback when FAISS is not installed

This keeps the API stable across environments while enabling performance upgrades automatically.

---

# Phase 4 Foundation (Audio Fingerprinting Engine)

## fingerprint_engine

Path: `mantra/fingerprint_engine/`

Responsibilities:

- generate Shazam-style fingerprints from audio (`generate_fingerprint`)
- detect high-energy spectrogram peaks
- construct landmark pairs `(freq1, freq2, delta_t)`
- hash landmarks into fingerprint keys for matching
- align query/database fingerprints by time-offset voting (`match_fingerprint`)

This enables robust matching under time shifts and supports future large-scale audio search.

---

# Phase 5 Foundation (HTTP API Layer)

## interfaces/api

Path: `mantra/interfaces/api/api_server.py`

Responsibilities:

- expose REST endpoints via FastAPI
- accept WAV uploads for indexing and search
- run audio feature extraction + embedding generation in request flow
- update and query the vector similarity index
- run fingerprint matching for snippet lookup

Endpoints:

- `POST /index/audio`
- `POST /search/audio`
- `POST /fingerprint/match`
- `GET /health`

This layer enables cloud-ready integration for clients, plugins, and external tools.

---

# Phase 6 Foundation (Persistent Music Database)

## database

Path: `mantra/database/`

Files:

- `schema.py`
- `track_store.py`

Responsibilities:

- persist indexed track metadata in SQLite
- store fingerprint payloads linked to track IDs
- provide CRUD operations for track records
- support API metadata enrichment during search/match responses
- coordinate persistent vector index path references (`embedding_path`)

Track metadata fields:

- `track_id`
- `filename`
- `duration`
- `embedding_path`
- `fingerprint_hash_count`
- `created_at`

This layer enables restart-safe indexing and stable metadata lookup for API clients.

---

# Phase 7 Foundation (Background Processing Pipeline)

## pipeline

Path: `mantra/pipeline/`

Files:

- `job_queue.py`
- `tasks.py`
- `worker.py`

Responsibilities:

- queue long-running indexing jobs (`enqueue`, `dequeue`)
- split processing into reusable tasks:
  - `embedding_task`
  - `fingerprint_task`
  - `index_audio_task`
- process queued jobs through worker loop (`BackgroundWorker`)

API integration:

- `POST /index/audio` enqueues indexing work instead of executing full indexing synchronously
- worker persists vector index snapshots, fingerprints, and SQLite metadata records

This decouples API latency from heavy feature extraction and indexing workloads.

---

# Phase 8 Foundation (Dataset Ingestion Pipeline)

## ingestion

Path: `mantra/ingestion/`

Files:

- `dataset_ingestor.py`
- `progress_tracker.py`

Responsibilities:

- scan large dataset folders for audio files (`scan_folder`)
- enqueue indexing jobs for every audio file (`enqueue_all_audio_files`)
- track ingestion execution progress:
  - `total_files`
  - `processed_files`
  - `failed_files`

API endpoints:

- `POST /ingest/dataset`
- `GET /ingest/status`

This enables bulk ingestion workflows for large music corpora while reusing the background worker pipeline.

---

# Phase 11-15: Scalable Audio Search Platform

## Target Project Structure

```
src/
  api/
  ingestion/
  search/
  indexing/
  workers/
```

Current implementation maps these domains under `mantra/` and includes a scaffolded `src/` layout for migration.

---

## Large-Scale Ingestion Architecture

Dataset ingestion flow for 10M tracks:

`dataset_folder -> scanner -> ingestion queue -> worker pool -> embedding/fingerprint tasks -> vector index update -> metadata store`

Key components:

- `mantra/ingestion/dataset_scanner.py` (`stream_audio_files`) for streaming discovery
- `mantra/ingestion/ingestion_queue.py` for queued ingestion jobs + retry metadata
- `mantra/ingestion/worker_pool.py` for parallel ingestion workers
- existing background pipeline (`mantra/pipeline/*`) for index-task execution

Supports:

- incremental ingestion
- resume-by-state (persistent DB + index artifacts)
- failure retry
- progress tracking (`total_files`, `processed_files`, `failed_files`)

---

## Worker Scaling Strategy

Worker model:

- API enqueues indexing jobs (non-blocking request path)
- dedicated workers consume jobs and persist outputs
- worker pool can be scaled horizontally (`docker compose` service replicas) or vertically (threads/processes)

Operational strategy for large datasets:

- shard dataset inputs across multiple scanners
- run multiple worker instances against shared queue backend
- use bounded retries and dead-letter logging for failed payloads

---

## Vector Indexing Strategy

Index backend:

- FAISS preferred when available
- numpy cosine fallback for environments without FAISS

Large-scale index behavior:

- incremental adds
- persistent save/load snapshots
- ANN-compatible API (`add`, `search`, `save`, `load`) via `vector_index.py`

This design supports migration from prototype-scale to multi-million vector collections while preserving API contracts.

---

## Search Request Lifecycle

Semantic search path:

1. validate query payload
2. load query audio
3. compute embedding
4. retrieve top-N candidates from vector index
5. apply metadata filters
6. rank + trim to `top_k`
7. return metadata-enriched results

Performance controls:

- batch search endpoint (`POST /search/batch`) for multi-query requests
- in-memory LRU result cache keyed by query-audio hash + params
- cache hit/miss metrics (`GET /search/cache/metrics`)

---

# Production Architecture (Phase 16-20)

## Architecture Diagram (Logical)

`API -> Queue -> Workers -> Embedding Engine -> Vector Index -> Storage`

Components:

- API: request handling, search, metadata filtering, batch orchestration, web UI serving
- Queue: in-memory/Redis pluggable backend for distributed workers
- Workers: parallel consumers with configurable concurrency and batching
- Vector Index: FAISS-preferred ANN index with persistent snapshots
- Storage: SQLite metadata + fingerprint records + index artifacts

---

## GPU Embedding Acceleration

- `mantra/embedding_engine.py` auto-detects GPU with `torch.cuda.is_available()`
- supports single and batch embedding computation
- workers can leverage batch embedding entry points for throughput improvements

---

## Distributed Queue + Workers

- `mantra/queue_backend.py`: in-memory default, optional Redis backend
- `mantra/ingestion/worker_pool.py`: `worker_id`, `concurrency`, `max_batch_size`
- supports independent worker processes polling queue backend

---

## Monitoring and Metrics

- `mantra/metrics.py` tracks:
  - ingestion_rate
  - embedding_latency
  - search_latency
  - cache_hit_rate
  - queue_depth
  - worker_failures
- `GET /metrics` exposes Prometheus-compatible text

---

## Web UI

- `web/index.html` provides basic upload + search interaction
- FastAPI serves UI route and static assets (`/web`)

---

## Dataset Manager

- `mantra/dataset_manager.py` tracks dataset registration and ingestion status
- endpoints:
  - `POST /dataset/register`
  - `POST /dataset/ingest`
  - `GET /dataset/list`
  - `GET /dataset/status`

---

# Enterprise Architecture (Phase 21-25)

## Cluster-Level Topology

`API Nodes -> Kafka/Queue Layer -> Worker Clusters -> Sharded FAISS Indexes -> Storage`

Additional services:

- Recommendation Engine (similarity graph and re-ranking)
- Metrics/Monitoring endpoint for platform observability

---

## Sharded Vector Search

- `sharded_index.py` manages multiple ANN shards.
- Strategies:
  - hash(track_id) (default)
  - range partition
  - round-robin
- Query fan-out across shards with merged top-k ranking.

---

## Kafka Streaming Ingestion

- `kafka_ingestion.py` provides producer/consumer flow.
- Dataset scanner publishes ingestion jobs to topic.
- Worker consumers process stream and update vector indexes incrementally.
- Fallback supported through existing Redis/InMemory queue paths.

---

## Online Index Updates

- `index_updater.py` supports real-time track insertion.
- API endpoint: `POST /index/add_track`.
- Supports incremental add + periodic compaction snapshots.
- Designed for concurrent search and insert operations.

---

## Recommendation + Re-Ranking

- `recommendation_engine.py` builds similarity graph and returns recommendations.
- Endpoint: `GET /recommend/{track_id}?k=10`.
- `reranker.py` applies final ranking from:
  - vector similarity
  - metadata match
  - popularity proxy

---

## ANN Routing

- `ann_router.py` adds centroid-based query routing using k-means.
- Queries are routed to top-N nearest centroid shards to reduce fan-out.

## Two-Stage Search

- `coarse_index.py` provides coarse cluster assignment.
- Search flow:
  1. coarse cluster selection
  2. candidate retrieval from selected clusters
  3. fine similarity ranking + reranking

## Vector Compression

- `vector_quantizer.py` provides product-quantization style encoding/decoding.
- Candidate vectors can be scored using decoded approximations for lower memory overhead.


## GPU FAISS Cluster

- `gpu_vector_index.py` adds GPU-aware ANN search.
- Loads CPU indexes and attempts GPU transfer (`to_gpu`).
- Falls back to CPU ANN transparently when GPU is unavailable.

## Feature Store

- `feature_store.py` persists track-level features for ranking/training.
- Supports single and batch feature retrieval.
- Designed for local SQLite persistence with easy migration path.

## ML Training Pipeline

- `training_pipeline.py` provides offline model training:
  - `train_embedding_model(dataset_path)`
  - `train_reranking_model(feature_store)`
  - `evaluate_models()`
- Outputs artifacts to `models/` for production loading.


## Model Registry

- `model_registry.py` manages versioned model artifacts.
- Supports registration, latest lookup, active version control.
- Storage layout: `models/<model_name>/<version>/` with registry metadata.

## A/B Testing

- `experiment_router.py` assigns users to experiment groups.
- Routes requests to model variants and logs experiment outcomes.
- API endpoints:
  - `GET /experiments`
  - `POST /experiments/create`

## Online Learning

- `online_learning.py` records user-track interaction events.
- Periodic `update_models()` aggregates feedback for continuous model updates.

## Autoscaling

- `autoscaler.py` consumes platform metrics and decides scale actions.
- Supports `monitor_metrics()`, `scale_up()`, `scale_down()`.
- Designed to scale worker pools based on queue depth/load/failure signals.


## Multimodal Embeddings

- `text_embedding_engine.py` adds text/lyrics embeddings.
- `multimodal_embedding.py` fuses audio + text + metadata vectors.

## Cross-Modal Search

- `cross_modal_search.py` supports text and lyrics query modes.
- API endpoints:
  - `POST /search/text`
  - `POST /search/lyrics`

## Playlist AI

- `playlist_generator.py` builds playlists from seed tracks and similarity graph recommendations.
- Endpoint: `GET /playlist/generate`.

## Conversational Assistant

- `music_assistant.py` parses user prompts into search/playlist intents.
- Endpoint: `POST /assistant/query`.


## Vector Index Service

- `vector_index_service.py` provides production index lifecycle APIs:
  - build/add/search/save/load
- Supports numpy-first implementation with optional FAISS backend via existing index adapters.

## Feature Store (Recommendation)

- `feature_store.py` now supports both user and track features:
  - user feature storage/retrieval
  - track feature storage/retrieval
- Enables ranking and home recommendation personalization.

## Session and Event Infrastructure

- `session_model.py` tracks per-user session state and short-term intent.
- `event_stream.py` provides threaded event ingestion + handler registration.
- API endpoint `/events` publishes real-time interaction signals.

## Home Recommendation APIs

- `/recommendations/home`
- `/recommendations/session`
- `/features/user`
- `/features/track`
- `/index/add`
- `/index/search`


## Graph Intelligence

- `graph_recommender.py` builds a user-track bipartite interaction graph.
- Personalized recommendations are computed with personalized PageRank when `networkx` is available, with a two-hop fallback path when it is not.
- Endpoint: `GET /recommendations/graph`.

## Trend and Viral Detection

- `trend_detector.py` consumes interaction events and computes:
  - trending tracks by frequency
  - track velocity in recent windows
  - viral track candidates
- `viral_predictor.py` estimates virality score from engagement features (`play_count`, `velocity`, `like_rate`) with optional sklearn logistic regression and a deterministic fallback heuristic.
- Endpoints:
  - `GET /tracks/trending`
  - `GET /tracks/viral`

## RL-Enhanced Ranking

- `rl_recommender.py` adds epsilon-greedy reinforcement policy with per-user Q-values.
- `/recommendations/home` pipeline:
  - candidate generation = vector index + graph recommender + trend detector
  - ranking = cosine ranker + RL reward bonus
- Endpoint for reward feedback: `POST /rl/reward`.

## Embedding Trainer Loop

- `embedding_trainer.py` trains and incrementally updates track embeddings from interaction streams.
- Event handlers trigger incremental updates after interaction events.
- On-demand training endpoint: `POST /embeddings/train`.

## Event-Driven Recommendation Pipeline

- Event flow:
  - `/events` -> `event_stream.py`
  - handlers update `trend_detector`, `feature_store`, `online_learning`, `embedding_trainer`, and session state.
- This keeps recommendation features, trend signals, and embedding updates synchronized with live user behavior.

## Transformer Session Modeling

- `transformer_session_model.py` provides sequence dataset creation, lightweight transformer-compatible training, and next-track prediction.
- Falls back to transition-frequency heuristics when PyTorch is unavailable.
- Session predictions are part of real-time candidate generation.

## User Taste Clustering

- `taste_clusterer.py` groups users by embedding similarity using sklearn KMeans, with deterministic hash-based fallback.
- Cluster assignments are used for recommendation boosts via cluster-preferred tracks.
- Endpoint: `GET /users/cluster`.

## Artist Intelligence

- `artist_intelligence.py` builds an artist similarity graph from track metadata (genre/year signals).
- Supports artist-to-artist similarity and user artist recommendation.
- Endpoint: `GET /artists/similar`.

## Realtime Recommendation Layer

- `realtime_recommender.py` combines:
  - vector index retrieval
  - graph recommender
  - transformer session model
  - trending tracks
- Endpoint: `GET /recommendations/realtime`.
- Home ranking applies:
  - cosine ranker
  - RL reward bonus
  - cluster preference boost

## Distributed Embedding Pipeline

- `distributed_embedding_pipeline.py` supports:
  - batch embedding training
  - incremental embedding updates
  - hash-based embedding sharding
- Endpoint: `POST /embeddings/batch-train`.

## Recommendation System Stats

- `GET /system/recommendation-stats` reports health signals for recommendation intelligence:
  - interaction count
  - clustering coverage
  - tracked embeddings
  - trending/viral candidate counts


## Phase 61-80: Graph Neural Recommender

- `gnn_recommender.py` adds a GraphSAGE/LightGCN-ready interface with fallback graph heuristics.
- User-track graph is trained from interaction streams and exposed via `GET /recommendations/gnn`.

## Phase 61-80: Foundation Audio Models

- `foundation_music_embeddings.py` provides foundation-style audio and multimodal embeddings (CLAP/AudioCLIP-style adapter surface).
- Deterministic fallback embeddings are used when foundation backends are unavailable.
- Embeddings can be cached into the feature store.

## Phase 61-80: Streaming Recommendation Engine

- `realtime_streaming_inference.py` subscribes to events and recomputes session-aware recommendations in real time.
- `realtime_feature_aggregator.py` maintains rolling play velocity, skip rate, and engagement score.
- Streaming status endpoint: `GET /system/streaming-status`.

## Phase 61-80: Bandit Optimization

- `bandit_recommender.py` supports Thompson Sampling, UCB, and epsilon-greedy policies.
- Reward updates are ingested through `POST /bandit/reward`.
- Bandit scores are fused into discovery feed ranking.

## Phase 61-80: AI DJ

- `ai_dj.py` creates persona-driven narrated playlist segments.
- Endpoint: `GET /ai-dj/session`.
- Service wrapper: `services/ai_dj_service.py`.

## Phase 61-80: Music Generation Pipeline

- `music_generation_pipeline.py` supports:
  - prompt-based generation
  - style transfer
  - loop generation
- Endpoints:
  - `POST /music/generate`
  - `POST /music/style-transfer`
- Service wrapper: `services/generation_service.py`.

## Phase 61-80: Discovery Feed

- `discovery_feed_ranker.py` fuses:
  - vector similarity
  - graph recommendation
  - GNN recommendation
  - bandit exploration
  - trending signal
  - cluster preference
  - engagement features
- Endpoint: `GET /recommendations/discovery`.

## Phase 61-80: Service Layer and Model Storage

- Added `services/streaming_inference_service.py`, `services/ai_dj_service.py`, `services/generation_service.py`.
- Added model storage structure:
  - `models/gnn_models/`
  - `models/foundation_embeddings/`
  - `models/generation_models/`


## Phase 81-120: Music Foundation Model

- `music_foundation_model.py` unifies embeddings for audio, lyrics, metadata, and artist-graph context.
- Supports multimodal embedding construction and foundation-model fine-tuning interface.

## Phase 81-120: Taste LLM

- `taste_llm.py` provides user taste summaries, next genre/artist predictions, and recommendation explanations.
- API:
  - `GET /ai/taste-profile`
  - `GET /ai/taste-explanation`

## Phase 81-120: Autonomous Optimization

- `autonomous_optimizer.py` monitors metrics and self-tunes ranking/exploration/cluster parameters.
- API: `GET /system/autonomous-optimization`.

## Phase 81-120: Global Trend Intelligence

- `global_trend_intelligence.py` detects global trends, clusters trending tracks, identifies emerging genres, and predicts viral growth.
- API: `GET /trends/global`.

## Phase 81-120: AI Music Agents

- `music_agent_system.py` coordinates discovery, playlist, trend, and generation agents.
- API: `GET /agents/status`.

## Phase 81-120: Generation Orchestrator

- `generation_orchestrator.py` orchestrates generation pipeline, remix, and advanced generation modes.
- APIs:
  - `POST /music/remix`
  - `POST /music/generate-advanced`

## Phase 81-120: Streaming Recommendation Orchestration

- `streaming_recommendation_orchestrator.py` dynamically chooses between vector/graph/GNN/bandit/transformer engines.
- Integrated with discovery feed responses and autonomous optimization status.

## Phase 81-120: Knowledge Graph

- `knowledge_graph_builder.py` builds global graph entities: users, artists, tracks, genres, playlists.
- Tracks relations including listens, likes, influence-style relationships.
- API: `GET /music/knowledge-graph`.

## Phase 81-120: AI Label Engine

- `ai_label_engine.py` simulates label operations:
  - new artist discovery
  - promotional playlist generation
  - fan growth tracking
- API support for breakout scouting through `GET /artists/breakout`.

## Phase 121200 Architecture

- Self-Evolving Recommendation System:
  - `self_evolving_recommender.py`
  - monitors model performance
  - auto retrain trigger
  - auto ranking strategy switching

- Meta Learning Engine:
  - `meta_learning_engine.py`
  - stores experiment outcomes
  - recommends best ranking pipeline

- Digital Twin Users:
  - `user_digital_twin.py`
  - simulates listening sessions and taste
  - evaluates recommendation policy changes

- Music Generation Lab:
  - `generative_music_lab.py`
  - runs multi-generator experiments

- Virtual Artist Factory:
  - `virtual_artist_factory.py`
  - creates AI artists, persona assignment, discography simulation

- Culture Trend Engine:
  - `culture_trend_engine.py`
  - detects cultural music shifts and genre evolution

- AI Producer System:
  - `ai_producer_engine.py`
  - melody/rhythm generation, stem mixing, mastering simulation

- Autonomous Experimentation:
  - `autonomous_experimentation.py`
  - automatic A/B launches, pipeline comparison, best-performer deployment

- AI Music Supervisor:
  - `ai_music_supervisor.py`
  - central coordination for recommenders and experiments

- Self-Improvement Loop:
  - `self_improvement_loop.py`
  - continuous metrics collection, retraining cycle, and model redeploy hooks

## Music Intelligence Layer

- Music Foundation Model:
  - `mantra/intelligence/music_foundation_model.py`
  - universal embeddings for audio, MIDI, and feature vectors
  - structure analysis outputs tempo, key, scale, harmony, rhythm, and spectral descriptors

- Music Genome System:
  - `mantra/intelligence/music_genome_engine.py`
  - structured genome extraction for genre, mood, energy, danceability, rhythm, harmony, melody, vocals, and production profile

- Genome Similarity Search:
  - `mantra/intelligence/music_genome_store.py`
  - vector-backed genome storage and nearest-neighbor retrieval with normalized similarity scoring

- Genome Conditioned Generation:
  - integrated in `ai_producer_engine.py` via `generate_track_from_genome`
  - conditions melody/rhythm/mix/mastering profile on genome attributes

- Platform Impact:
  - Recommendation: genome + foundation signals enhance track relevance
  - Discovery: additional ranking signals for genome similarity and foundation embedding similarity
  - Generation: controllable synthesis guided by music genome descriptors
  - Trend detection: shared genome attributes support ecosystem-level musical pattern tracking

