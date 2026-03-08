"""Global trend intelligence for tracks/genres and viral growth estimation."""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from typing import Dict, Iterable, List


class GlobalTrendIntelligence:
    def __init__(self):
        self.events: List[Dict[str, object]] = []

    def ingest(self, events: Iterable[Dict[str, object]]) -> None:
        for event in events:
            payload = dict(event)
            payload.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
            self.events.append(payload)

    def _window(self, hours: int) -> List[Dict[str, object]]:
        now = datetime.now(timezone.utc)
        start = now - timedelta(hours=max(1, int(hours)))
        out = []
        for e in self.events:
            try:
                ts = datetime.fromisoformat(str(e.get("timestamp")))
            except Exception:
                continue
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            if ts >= start:
                out.append(e)
        return out

    def detect_global_trends(self, top_k: int = 20) -> List[Dict[str, object]]:
        counts = Counter(str(e.get("track_id") or "") for e in self.events if e.get("track_id"))
        return [{"track_id": tid, "count": c} for tid, c in counts.most_common(max(0, int(top_k)))]

    def cluster_trending_tracks(self, top_k: int = 50) -> Dict[str, List[str]]:
        clusters: Dict[str, List[str]] = defaultdict(list)
        for item in self.detect_global_trends(top_k=top_k):
            bucket = f"cluster_{abs(hash(item['track_id'])) % 5}"
            clusters[bucket].append(str(item["track_id"]))
        return dict(clusters)

    def identify_emerging_genres(self, top_k: int = 10) -> List[Dict[str, object]]:
        recent = self._window(6)
        baseline = self._window(48)
        recent_counts = Counter(str(e.get("genre") or "") for e in recent if e.get("genre"))
        baseline_counts = Counter(str(e.get("genre") or "") for e in baseline if e.get("genre"))
        scored = []
        for genre, rc in recent_counts.items():
            b = baseline_counts.get(genre, 0)
            growth = float(rc) - float(b) / 8.0
            scored.append({"genre": genre, "growth": growth})
        scored.sort(key=lambda x: x["growth"], reverse=True)
        return scored[: max(0, int(top_k))]

    def predict_viral_growth(self, track_id: str) -> float:
        recent = [e for e in self._window(2) if str(e.get("track_id")) == str(track_id)]
        baseline = [e for e in self._window(24) if str(e.get("track_id")) == str(track_id)]
        return float(len(recent) - (len(baseline) / 12.0))

