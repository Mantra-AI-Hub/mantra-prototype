"""Kafka-style streaming ingestion with fallback backend."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, Optional


@dataclass
class KafkaMessage:
    key: str
    value: Dict[str, Any]


class MockKafkaBroker:
    def __init__(self):
        self._topics: Dict[str, Deque[KafkaMessage]] = {}

    def publish(self, topic: str, key: str, value: Dict[str, Any]) -> None:
        self._topics.setdefault(topic, deque()).append(KafkaMessage(key=key, value=dict(value)))

    def poll(self, topic: str) -> Optional[KafkaMessage]:
        queue = self._topics.setdefault(topic, deque())
        if not queue:
            return None
        return queue.popleft()


class KafkaIngestionPipeline:
    """Producer/consumer facade with optional kafka-python backend."""

    def __init__(self, topic: str = "mantra.ingestion", broker: MockKafkaBroker | None = None):
        self.topic = topic
        self.broker = broker or MockKafkaBroker()
        self.backend = "mock"

    def produce(self, key: str, payload: Dict[str, Any]) -> None:
        self.broker.publish(self.topic, key=key, value=payload)

    def consume(self) -> Optional[KafkaMessage]:
        return self.broker.poll(self.topic)

    def consume_one(self, handler) -> bool:
        message = self.consume()
        if message is None:
            return False
        return bool(handler(message))
