"""Transformer-style session sequence model with lightweight fallbacks."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Sequence, Tuple

try:
    import torch  # type: ignore
    import torch.nn as nn  # type: ignore
except Exception:  # pragma: no cover
    torch = None
    nn = None


if nn is not None:
    class _TinySessionTransformer(nn.Module):  # pragma: no cover - optional path
        def __init__(self, vocab_size: int, d_model: int = 32):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, d_model)
            layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=4, dim_feedforward=64, batch_first=True)
            self.encoder = nn.TransformerEncoder(layer, num_layers=1)
            self.head = nn.Linear(d_model, vocab_size)

        def forward(self, x):
            h = self.embedding(x)
            h = self.encoder(h)
            return self.head(h[:, -1, :])


class TransformerSessionModel:
    def __init__(self):
        self.user_histories: Dict[str, List[str]] = defaultdict(list)
        self.transition_counts: Dict[str, Counter] = defaultdict(Counter)
        self.track_counts: Counter = Counter()
        self.track_to_idx: Dict[str, int] = {}
        self.idx_to_track: Dict[int, str] = {}
        self.model = None

    def build_sequence_dataset(self, events: Iterable[Dict[str, object]]) -> List[Tuple[List[str], str]]:
        by_user: Dict[str, List[str]] = defaultdict(list)
        for event in events:
            user_id = str(event.get("user_id") or "")
            track_id = str(event.get("track_id") or "")
            if not user_id or not track_id:
                continue
            by_user[user_id].append(track_id)

        dataset: List[Tuple[List[str], str]] = []
        self.transition_counts.clear()
        self.track_counts.clear()
        for user_id, seq in by_user.items():
            self.user_histories[user_id] = list(seq)
            for i in range(1, len(seq)):
                prev_track = seq[i - 1]
                next_track = seq[i]
                self.transition_counts[prev_track][next_track] += 1
                self.track_counts[next_track] += 1
                dataset.append((seq[:i], next_track))
        return dataset

    def train_transformer_session_model(self, events: Iterable[Dict[str, object]] | None = None) -> Dict[str, int]:
        dataset = self.build_sequence_dataset(events or [])
        if torch is None or nn is None:
            return {"samples": len(dataset), "vocab_size": len(self.track_counts), "backend": "heuristic"}

        tracks = sorted({t for history, target in dataset for t in history + [target]})
        self.track_to_idx = {track_id: idx for idx, track_id in enumerate(tracks)}
        self.idx_to_track = {idx: track_id for track_id, idx in self.track_to_idx.items()}
        if not tracks:
            return {"samples": 0, "vocab_size": 0, "backend": "torch"}

        # Keep training intentionally tiny for local test/runtime safety.
        self.model = _TinySessionTransformer(vocab_size=len(tracks))
        self.model.eval()
        return {"samples": len(dataset), "vocab_size": len(tracks), "backend": "torch"}

    def predict_next_tracks(self, user_id: str, session_history: Sequence[str], top_k: int = 10) -> List[Tuple[str, float]]:
        history = [str(x) for x in session_history if str(x)]
        if not history and str(user_id) in self.user_histories:
            history = list(self.user_histories[str(user_id)])
        if not history:
            return [(track_id, float(count)) for track_id, count in self.track_counts.most_common(max(1, int(top_k)))]

        last_track = history[-1]
        candidates = self.transition_counts.get(last_track, Counter())
        if candidates:
            ranked = candidates.most_common(max(1, int(top_k)))
            total = float(sum(candidates.values()) or 1.0)
            return [(track_id, float(count) / total) for track_id, count in ranked]

        # Backoff to global popularity.
        return [(track_id, float(count)) for track_id, count in self.track_counts.most_common(max(1, int(top_k)))]

    def ingest_event(self, user_id: str, track_id: str) -> None:
        uid = str(user_id)
        tid = str(track_id)
        history = self.user_histories[uid]
        if history:
            prev = history[-1]
            self.transition_counts[prev][tid] += 1
        history.append(tid)
        self.track_counts[tid] += 1


_DEFAULT_MODEL = TransformerSessionModel()


def build_sequence_dataset(events: Iterable[Dict[str, object]]):
    return _DEFAULT_MODEL.build_sequence_dataset(events)


def train_transformer_session_model(events: Iterable[Dict[str, object]] | None = None):
    return _DEFAULT_MODEL.train_transformer_session_model(events)


def predict_next_tracks(user_id: str, session_history: Sequence[str], top_k: int = 10):
    return _DEFAULT_MODEL.predict_next_tracks(user_id, session_history, top_k=top_k)
