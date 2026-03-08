"""Distributed-style embedding pipeline helpers."""

from __future__ import annotations

from typing import Dict, Iterable, List


class DistributedEmbeddingPipeline:
    def __init__(self, embedding_trainer):
        self.embedding_trainer = embedding_trainer
        self.last_shards: Dict[int, List[str]] = {}

    def batch_train_embeddings(self, interactions: Iterable[Dict[str, object]]) -> Dict[str, object]:
        embeddings = self.embedding_trainer.train_track_embeddings(list(interactions))
        path = self.embedding_trainer.save_embeddings()
        return {"tracks": len(embeddings), "path": path}

    def incremental_update_embeddings(self, interactions: Iterable[Dict[str, object]]) -> Dict[str, object]:
        embeddings = self.embedding_trainer.update_embeddings_incrementally(list(interactions))
        return {"tracks": len(embeddings), "path": self.embedding_trainer.output_path}

    def shard_embeddings(self, num_shards: int = 4) -> Dict[int, List[str]]:
        total_shards = max(1, int(num_shards))
        shards: Dict[int, List[str]] = {idx: [] for idx in range(total_shards)}
        for track_id in self.embedding_trainer.track_embeddings.keys():
            shard_id = int(abs(hash(str(track_id))) % total_shards)
            shards[shard_id].append(str(track_id))
        self.last_shards = shards
        return shards


