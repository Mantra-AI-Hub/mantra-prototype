from mantra.embedding_trainer import EmbeddingTrainer
from mantra.distributed_embedding_pipeline import DistributedEmbeddingPipeline


def test_distributed_embedding_pipeline_train_update_and_shard():
    trainer = EmbeddingTrainer(output_path="test_distributed_embeddings.npy", dim=8)
    pipeline = DistributedEmbeddingPipeline(embedding_trainer=trainer)

    batch = pipeline.batch_train_embeddings(
        [
            {"track_id": "t1", "reward": 1.0},
            {"track_id": "t2", "reward": 2.0},
        ]
    )
    assert batch["tracks"] == 2

    inc = pipeline.incremental_update_embeddings([{"track_id": "t3", "reward": 1.0}])
    assert inc["tracks"] >= 2

    shards = pipeline.shard_embeddings(num_shards=3)
    assert len(shards) == 3
    assert sum(len(v) for v in shards.values()) >= 2

