"""Lightweight virality predictor for tracks."""

from __future__ import annotations

from typing import Dict

import numpy as np

try:
    from sklearn.linear_model import LogisticRegression  # type: ignore
except Exception:  # pragma: no cover
    LogisticRegression = None


class ViralPredictor:
    def __init__(self):
        self.model = None
        self.fallback_weights = np.array([0.4, 0.3, 0.3], dtype=np.float32)

    def _feature_vector(self, features: Dict[str, object]) -> np.ndarray:
        return np.array(
            [
                float(features.get("play_count", 0.0)),
                float(features.get("velocity", 0.0)),
                float(features.get("like_rate", 0.0)),
            ],
            dtype=np.float32,
        )

    def fit(self, dataset: list[tuple[Dict[str, object], int]]) -> None:
        if LogisticRegression is None or not dataset:
            return
        x = np.vstack([self._feature_vector(f) for f, _ in dataset])
        y = np.array([int(v) for _, v in dataset], dtype=np.int32)
        self.model = LogisticRegression(max_iter=200)
        self.model.fit(x, y)

    def predict_track_virality(self, track_features: Dict[str, object]) -> float:
        vec = self._feature_vector(track_features)
        if self.model is not None:
            proba = self.model.predict_proba(vec.reshape(1, -1))[0][1]
            return float(proba)

        # Fallback heuristic
        normalized = np.tanh(vec / np.array([1000.0, 20.0, 1.0], dtype=np.float32))
        score = float(np.clip(np.dot(normalized, self.fallback_weights), 0.0, 1.0))
        return score
