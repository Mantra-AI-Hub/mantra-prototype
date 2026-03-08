"""Sharded vector index for horizontal scaling."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from mantra.vector_index import VectorIndex


class ShardedVectorIndex:
    def __init__(
        self,
        dimension: int,
        shard_count: int = 4,
        strategy: str = "hash",
    ):
        if shard_count <= 0:
            raise ValueError("shard_count must be > 0")
        self.dimension = int(dimension)
        self.shard_count = int(shard_count)
        self.strategy = str(strategy)
        self._shards = [VectorIndex(dimension=self.dimension) for _ in range(self.shard_count)]
        self._rr_counter = 0

    def _choose_shard(self, track_id: str) -> int:
        if self.strategy == "hash":
            return hash(track_id) % self.shard_count
        if self.strategy == "range":
            # lightweight range-style partition based on first character
            first = ord(track_id[0]) if track_id else 0
            bucket = max(1, 256 // self.shard_count)
            return min(self.shard_count - 1, first // bucket)
        if self.strategy == "round-robin":
            index = self._rr_counter % self.shard_count
            self._rr_counter += 1
            return index
        return hash(track_id) % self.shard_count

    def add(self, vector: Sequence[float], track_id: str) -> None:
        shard_idx = self._choose_shard(str(track_id))
        self._shards[shard_idx].add(vector=vector, track_id=track_id)

    def search(self, vector: Sequence[float], k: int) -> List[Tuple[str, float]]:
        merged: List[Tuple[str, float]] = []
        for shard in self._shards:
            merged.extend(shard.search(vector=vector, k=max(k, k * 2)))
        merged.sort(key=lambda item: item[1], reverse=True)
        return merged[: max(0, int(k))]

    def save(self, path: str) -> None:
        base = Path(path)
        base.parent.mkdir(parents=True, exist_ok=True)
        meta = {
            "dimension": self.dimension,
            "shard_count": self.shard_count,
            "strategy": self.strategy,
        }
        (base.with_suffix(base.suffix + ".meta.json")).write_text(json.dumps(meta), encoding="utf-8")
        for idx, shard in enumerate(self._shards):
            shard.save(str(base.with_suffix(base.suffix + f".shard{idx}")))

    @classmethod
    def load(cls, path: str) -> "ShardedVectorIndex":
        base = Path(path)
        meta_path = base.with_suffix(base.suffix + ".meta.json")
        if not meta_path.exists():
            raise FileNotFoundError(f"No sharded index metadata found: {meta_path}")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        obj = cls(
            dimension=int(meta["dimension"]),
            shard_count=int(meta["shard_count"]),
            strategy=str(meta.get("strategy", "hash")),
        )
        shards = []
        for idx in range(obj.shard_count):
            shard = VectorIndex.load(str(base.with_suffix(base.suffix + f".shard{idx}")))
            shards.append(shard)
        obj._shards = shards
        return obj
