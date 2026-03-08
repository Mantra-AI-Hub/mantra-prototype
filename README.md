# 🎵 Mantra Prototype

Structural MIDI Similarity Engine for Music Analysis & IP Protection

Mantra is an interval-based music fingerprinting engine designed to detect structural melodic similarity between MIDI sequences.

---

# 🚀 Core Value

Traditional tools compare audio waveforms.  
Mantra compares melodic structure.

✔ Key-invariant  
✔ Tempo-independent  
✔ Structure-aware  
✔ Regression-tested  

---

# 🧠 Technical Architecture

Pipeline:

MIDI → Pitch Extraction → Interval Normalization → Fingerprint → Similarity Score

---

# 📦 Installation

Create environment:

Windows:

python -m venv venv  
venv\Scripts\activate  

Install dependencies:

pip install -r requirements.txt

---

# 🧪 Run Tests

pytest

Expected result:

14 passed

---

# 🌐 Run API

uvicorn app.main:app --reload

Open browser:

http://127.0.0.1:8000/docs

---

# 📊 Current Status

✔ Stable similarity engine  
✔ Regression validation  
✔ Modular architecture  
✔ SQLite persistence  
✔ FastAPI integration  

---

# 💼 Commercial Direction

• Plagiarism detection SaaS  
• Composer similarity analytics  
• Copyright verification API  
• DAW plugin integration  
• Label & publisher tools  

---

# 📜 License

MIT License
---

# CLI Usage

Run CLI commands via module:

python -m mantra.interfaces.cli.mantra_cli index <folder> --db-path fingerprints.db  
python -m mantra.interfaces.cli.mantra_cli search <midi_file> --db-path fingerprints.db  
python -m mantra.interfaces.cli.mantra_cli explain <midi_a> <midi_b>  

Commands:

- `index`: scan dataset folder and index MIDI fingerprints into SQLite
- `search`: find similar tracks using MinHash + LSH + SQLite signatures
- `explain`: return interval, pitch, and rhythm-based similarity explanation

---

# New Modules (Phase 1)

- `mantra/dataset_engine/`: dataset scanning, MIDI loading, melody/pitch/interval extraction
- `mantra/explain_engine/`: structured similarity explanation engine
- `mantra/interfaces/cli/`: argparse-based CLI (`index`, `search`, `explain`)

---

# Mantra Doctor

System diagnostic CLI for health checks across imports, recommendation, generation, vector index, feature store, models, supervisor, experiments, self-evolving recommender, and API module availability.

Example usage:

python -m mantra.doctor
python -m mantra.doctor --metrics
python -m mantra.doctor --json

---

# Production Deployment (Phase 9)

## Docker

Build image:

docker build -t mantra:latest .

Run API container:

docker run --rm -p 8000:8000 -v mantra_data:/app/data mantra:latest

## Docker Compose

Start API + worker with shared persistent data volume:

docker compose up --build

Services:

- `api`: FastAPI server on `http://127.0.0.1:8000`
- `worker`: background queue worker for indexing jobs
- `mantra_data` volume: shared `/app/data` for SQLite + vector index artifacts

## Local Scripts

Run API:

python scripts/run_api.py

Run worker:

python scripts/run_worker.py

---

# Semantic Search API

Endpoint:

`POST /search`

Request body:

```json
{
  "query_audio_path": "path/to/file.wav",
  "top_k": 5
}
```

Behavior:

- loads query WAV from disk
- computes query embedding with the same feature + embedding pipeline used during indexing
- searches persisted vector index
- returns nearest tracks with score + metadata

Response:

```json
{
  "results": [
    {
      "track_id": "...",
      "score": 0.93,
      "metadata": {
        "track_id": "...",
        "filename": "...",
        "duration": 1.0,
        "embedding_path": "...",
        "fingerprint_hash_count": 120,
        "created_at": "..."
      }
    }
  ]
}
```

---

# Search API

Semantic search endpoint:

`POST /search`

Request:

```json
{
  "query_audio_path": "path/to/file.wav",
  "top_k": 10,
  "filters": {
    "genre": "ambient",
    "artist": "unknown"
  }
}
```

Response:

```json
{
  "results": [
    {
      "track_id": "...",
      "score": 0.93,
      "metadata": { "genre": "ambient", "artist": "unknown" }
    }
  ]
}
```

---

# Batch Search API

Batch semantic search endpoint:

`POST /search/batch`

Request:

```json
{
  "queries": [
    {"query_audio_path": "path/to/q1.wav"},
    {"query_audio_path": "path/to/q2.wav", "filters": {"genre": "ambient"}}
  ],
  "top_k": 5
}
```

Response returns a result list per query.

---

# Dataset Ingestion Pipeline

Endpoints:

- `POST /ingest/dataset?path=<folder>`
- `GET /ingest/status`

Pipeline:

`scanner -> queue -> workers -> embedding/fingerprint -> vector index + metadata DB`

---

# Running Distributed Workers

Local worker process:

`python scripts/run_worker.py`

Docker Compose:

`docker compose up --build`

Scale workers (example):

`docker compose up --build --scale worker=4`

---

# Scaling To Millions of Tracks

Recommended strategy:

1. Use FAISS backend for ANN search.
2. Keep index and SQLite metadata on persistent volumes.
3. Run multiple worker instances for ingestion throughput.
4. Use batch search and cache metrics to tune request cost.
5. Shard ingestion sources and process incrementally with retries.

---

# GPU Acceleration

`mantra/embedding_engine.py` provides optional GPU acceleration.

- Auto-detects CUDA via `torch.cuda.is_available()`.
- Supports:
  - `compute_embedding(audio_path)`
  - `compute_embeddings_batch(audio_paths)`

---

# Distributed Workers

Queue backends in `mantra/queue_backend.py`:

- `InMemoryQueueBackend` (default)
- `RedisQueueBackend` (optional)

Worker pool supports:

- `worker_id`
- `concurrency`
- `max_batch_size`

---

# Metrics and Monitoring

Prometheus-compatible endpoint:

`GET /metrics`

Tracks:

- ingestion_rate
- embedding_latency
- search_latency
- cache_hit_rate
- queue_depth
- worker_failures

---

# Web UI

Open:

`GET /web`

Simple browser UI lets you upload WAV and run similarity search.

---

# Dataset Manager

Endpoints:

- `POST /dataset/register`
- `POST /dataset/ingest`
- `GET /dataset/list`
- `GET /dataset/status`

Dataset fields:

- dataset_id
- name
- track_count
- ingestion_status

---

# Sharded Vector Search

Use `sharded_index.py` to scale vector retrieval horizontally.

- shard strategies: `hash`, `range`, `round-robin`
- fan-out query, merge top-k globally

---

# Kafka Streaming Ingestion

`kafka_ingestion.py` provides streaming producer/consumer primitives.

Flow:

`scanner -> Kafka topic -> consumer workers -> embeddings -> vector index updates`

Fallbacks:

- Redis queue backend
- In-memory queue backend

---

# Online Index Updates

Endpoint:

`POST /index/add_track`

Request body:

```json
{
  "audio_path": "path/to/file.wav",
  "metadata": {
    "track_id": "...",
    "genre": "ambient"
  }
}
```

---

# Recommendation API

Endpoint:

`GET /recommend/{track_id}?k=10`

Returns graph-based nearest recommendations with metadata.

---

# Re-ranking Search Results

Search now supports post-retrieval re-ranking using `reranker.py`:

`final_score = similarity*0.8 + metadata_match*0.1 + popularity*0.1`
