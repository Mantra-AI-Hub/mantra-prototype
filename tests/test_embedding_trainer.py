from pathlib import Path
from uuid import uuid4

from mantra.embedding_trainer import EmbeddingTrainer


def test_embedding_trainer_train_incremental_and_save():
    output_path = f"test_models_{uuid4().hex}.npy"
    trainer = EmbeddingTrainer(output_path=output_path, dim=16)

    embeddings = trainer.train_track_embeddings(
        [
            {"track_id": "t1", "reward": 1.0},
            {"track_id": "t1", "reward": 2.0},
            {"track_id": "t2", "reward": 1.0},
        ]
    )
    assert set(embeddings.keys()) == {"t1", "t2"}

    updated = trainer.update_embeddings_incrementally([{"track_id": "t1", "reward": 1.0}])
    assert "t1" in updated

    saved = trainer.save_embeddings()
    assert Path(saved).exists()
    assert Path(saved + ".ids").exists()

