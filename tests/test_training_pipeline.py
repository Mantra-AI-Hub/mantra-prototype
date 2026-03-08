from pathlib import Path
from uuid import uuid4

import numpy as np

from mantra.feature_store import FeatureStore
from mantra.training_pipeline import evaluate_models, train_embedding_model, train_reranking_model


def test_training_pipeline_outputs_models_and_metrics():
    run_id = uuid4().hex

    dataset = Path(f"test_training_vectors_{run_id}.npy")
    vectors = np.random.randn(20, 8).astype(np.float32)
    np.save(dataset, vectors)

    store = FeatureStore(db_path=f"test_training_store_{run_id}.db")
    store.put("t1", {"popularity": 0.2})
    store.put("t2", {"popularity": 0.4})

    emb_model_path = train_embedding_model(str(dataset))
    rerank_model_path = train_reranking_model(store)
    metrics = evaluate_models()

    assert Path(emb_model_path).exists()
    assert Path(rerank_model_path).exists()
    assert metrics["embedding_model_available"] == 1.0
    assert metrics["reranking_model_available"] == 1.0
    assert metrics["reranking_weight_sum"] > 0
