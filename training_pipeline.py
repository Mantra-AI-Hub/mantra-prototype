"""Offline training pipeline for embedding and reranking models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np

from mantra.feature_store import FeatureStore


_MODELS_DIR = Path("models")


def _ensure_models_dir() -> None:
    _MODELS_DIR.mkdir(parents=True, exist_ok=True)


def train_embedding_model(dataset_path: str) -> str:
    """Train a lightweight embedding projection from dataset vectors."""
    _ensure_models_dir()

    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    vectors = np.load(path)
    if vectors.ndim != 2:
        raise ValueError("Embedding dataset must be 2D")

    mean = vectors.mean(axis=0)
    std = vectors.std(axis=0) + 1e-8

    model_path = _MODELS_DIR / "embedding_model.npz"
    np.savez(model_path, mean=mean.astype(np.float32), std=std.astype(np.float32))
    return str(model_path)


def train_reranking_model(feature_store: FeatureStore) -> str:
    """Train a lightweight reranking weight model from feature statistics."""
    _ensure_models_dir()

    items = feature_store.all_items()
    if not items:
        weights = {"similarity": 0.8, "metadata_match": 0.1, "popularity": 0.1}
    else:
        popularity_vals = []
        for _, feats in items:
            popularity_vals.append(float(feats.get("popularity", 0.0)))
        pop_mean = float(np.mean(popularity_vals)) if popularity_vals else 0.0
        weights = {
            "similarity": 0.75,
            "metadata_match": 0.1,
            "popularity": min(0.15, max(0.05, pop_mean)),
        }

    path = _MODELS_DIR / "reranking_model.json"
    path.write_text(json.dumps(weights), encoding="utf-8")
    return str(path)


def evaluate_models() -> Dict[str, float]:
    """Evaluate availability and basic quality checks for trained models."""
    embedding_model = _MODELS_DIR / "embedding_model.npz"
    reranking_model = _MODELS_DIR / "reranking_model.json"

    metrics = {
        "embedding_model_available": 1.0 if embedding_model.exists() else 0.0,
        "reranking_model_available": 1.0 if reranking_model.exists() else 0.0,
    }

    if embedding_model.exists():
        payload = np.load(embedding_model)
        mean = payload["mean"]
        std = payload["std"]
        metrics["embedding_model_quality"] = float(np.mean(np.abs(mean)) / (np.mean(std) + 1e-6))
    else:
        metrics["embedding_model_quality"] = 0.0

    if reranking_model.exists():
        weights = json.loads(reranking_model.read_text(encoding="utf-8"))
        metrics["reranking_weight_sum"] = float(sum(float(v) for v in weights.values()))
    else:
        metrics["reranking_weight_sum"] = 0.0

    return metrics
