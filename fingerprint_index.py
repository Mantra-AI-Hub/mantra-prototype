"""Persistent inverted index for audio fingerprints."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, List, Tuple


Fingerprint = Tuple[str, int]


class FingerprintIndex:
    def __init__(self):
        self._index: DefaultDict[str, List[Tuple[str, int]]] = defaultdict(list)

    def add(self, track_id: str, fingerprints: Iterable[Fingerprint]) -> None:
        for hash_key, offset in fingerprints:
            self._index[str(hash_key)].append((str(track_id), int(offset)))

    def query(self, fingerprints: Iterable[Fingerprint]) -> List[Dict[str, object]]:
        alignment: DefaultDict[Tuple[str, int], int] = defaultdict(int)
        for hash_key, query_offset in fingerprints:
            for track_id, track_offset in self._index.get(str(hash_key), []):
                delta = int(track_offset) - int(query_offset)
                alignment[(track_id, delta)] += 1

        by_track: DefaultDict[str, Tuple[int, int]] = defaultdict(lambda: (0, 0))
        for (track_id, delta), count in alignment.items():
            best_count, best_delta = by_track[track_id]
            if count > best_count:
                by_track[track_id] = (count, delta)

        total = max(1, sum(count for count, _ in by_track.values()))
        results: List[Dict[str, object]] = []
        for track_id, (count, delta) in by_track.items():
            results.append(
                {
                    "track_id": track_id,
                    "score": float(count / total),
                    "offset": float(delta),
                }
            )
        results.sort(key=lambda item: item["score"], reverse=True)
        return results

    def save(self, path: str) -> None:
        serializable = {key: value for key, value in self._index.items()}
        Path(path).write_text(json.dumps(serializable), encoding="utf-8")

    @classmethod
    def load(cls, path: str) -> "FingerprintIndex":
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Fingerprint index not found: {path}")
        raw = json.loads(p.read_text(encoding="utf-8"))
        obj = cls()
        for hash_key, values in raw.items():
            obj._index[str(hash_key)] = [(str(item[0]), int(item[1])) for item in values]
        return obj
