from mantra.queue_backend import InMemoryQueueBackend


def test_inmemory_queue_backend_distributed_contract():
    queue = InMemoryQueueBackend()
    j1 = queue.enqueue({"task": "a"})
    j2 = queue.enqueue({"task": "b"})

    first = queue.dequeue()
    second = queue.dequeue()

    assert first is not None and first.job_id == j1.job_id
    assert second is not None and second.job_id == j2.job_id
    queue.ack(first)
    queue.ack(second)
    assert queue.depth() == 0
