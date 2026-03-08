from mantra.kafka_ingestion import KafkaIngestionPipeline, MockKafkaBroker


def test_kafka_ingestion_mock_pipeline():
    broker = MockKafkaBroker()
    pipeline = KafkaIngestionPipeline(topic="t1", broker=broker)

    pipeline.produce("id-1", {"track": "a"})
    pipeline.produce("id-2", {"track": "b"})

    seen = []

    def handler(message):
        seen.append(message.value["track"])
        return True

    assert pipeline.consume_one(handler)
    assert pipeline.consume_one(handler)
    assert not pipeline.consume_one(handler)
    assert seen == ["a", "b"]
