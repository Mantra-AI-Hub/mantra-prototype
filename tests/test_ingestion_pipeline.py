import shutil
import time
from pathlib import Path
from uuid import uuid4

from mantra.ingestion.dataset_scanner import stream_audio_files
from mantra.ingestion.ingestion_queue import IngestionQueue
from mantra.ingestion.worker_pool import WorkerPool


def test_stream_audio_files_and_worker_pool_retry():
    root = Path(f"test_ingestion_pipeline_{uuid4().hex}")
    root.mkdir(parents=True, exist_ok=True)
    try:
        (root / "a.wav").write_bytes(b"RIFF")
        (root / "b.wav").write_bytes(b"RIFF")
        (root / "ignore.txt").write_text("x", encoding="utf-8")

        files = list(stream_audio_files(str(root)))
        assert len(files) == 2

        queue = IngestionQueue()
        attempts = {"job": 0}

        def handler(job):
            attempts["job"] += 1
            return attempts["job"] > 1

        queue.enqueue("index_audio", {"id": 1})
        pool = WorkerPool(queue=queue, job_handler=handler, workers=1, max_retries=2, poll_interval=0.01)
        pool.start()
        time.sleep(0.2)
        pool.stop()

        assert attempts["job"] >= 2
    finally:
        shutil.rmtree(root, ignore_errors=True)
