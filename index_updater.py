"""Online index updates for real-time track additions."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict

from mantra.embedding_engine import EmbeddingEngine
from mantra.feature_store import FeatureStore
from mantra.intelligence.music_foundation_model import MusicFoundationModel
from mantra.intelligence.music_genome_engine import MusicGenomeEngine
from mantra.intelligence.music_genome_store import MusicGenomeStore
from mantra.search_engine import clear_search_cache


class IndexUpdater:
    def __init__(self, vector_index, vector_index_path: str, track_store, feature_store: FeatureStore | None = None):
        self.vector_index = vector_index
        self.vector_index_path = vector_index_path
        self.track_store = track_store
        self.feature_store = feature_store
        self.embedding_engine = EmbeddingEngine()
        self.embedding_engine.load_model()
        self.foundation_model = MusicFoundationModel()
        self.genome_engine = MusicGenomeEngine()
        self.genome_store = MusicGenomeStore()

    def add_track(self, audio_path: str, metadata: Dict[str, object]) -> Dict[str, object]:
        track_id = str(metadata.get("track_id") or metadata.get("filename") or audio_path)
        embedding = self.embedding_engine.compute_embedding(audio_path)
        foundation_embedding = self.foundation_model.embed_audio(audio_path).tolist()
        structure = self.foundation_model.analyze_structure(audio_path)
        genome = self.genome_engine.extract_genome(audio_path)
        self.genome_store.store_genome(track_id, genome)
        try:
            self.vector_index.add(vector=embedding, track_id=track_id)
        except TypeError:
            # Backward compatibility with older index signature add(track_id, embedding).
            self.vector_index.add(track_id, embedding)
        self.vector_index.save(self.vector_index_path)

        record = {
            "track_id": track_id,
            "filename": str(metadata.get("filename") or audio_path),
            "duration": float(metadata.get("duration") or 0.0),
            "embedding_path": self.vector_index_path,
            "fingerprint_hash_count": int(metadata.get("fingerprint_hash_count") or 0),
            "created_at": str(metadata.get("created_at") or datetime.now(timezone.utc).isoformat()),
            "artist": str(metadata.get("artist") or ""),
            "album": str(metadata.get("album") or ""),
            "genre": str(metadata.get("genre") or ""),
            "tags": metadata.get("tags") or [],
            "year": metadata.get("year"),
            "foundation_embedding": foundation_embedding,
            "music_structure": structure,
            "music_genome": genome,
        }
        self.track_store.add_track(record)
        if self.feature_store is not None:
            self.feature_store.store_track_features(
                track_id,
                {
                    "foundation_embedding": foundation_embedding,
                    "music_structure": structure,
                    "music_genome": genome,
                },
            )
        clear_search_cache()
        return record

    def compact(self) -> None:
        # Lightweight compaction placeholder: persist snapshot atomically.
        self.vector_index.save(self.vector_index_path)
