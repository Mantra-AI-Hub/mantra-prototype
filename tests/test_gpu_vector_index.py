from uuid import uuid4

import numpy as np

from mantra.gpu_vector_index import GPUVectorIndex
from mantra.vector_index import VectorIndex


def test_gpu_vector_index_load_search_and_add():
    base = f"test_gpu_index_{uuid4().hex}"

    cpu = VectorIndex(dimension=4)
    cpu.add([1.0, 0.0, 0.0, 0.0], "a")
    cpu.add([0.0, 1.0, 0.0, 0.0], "b")
    cpu.save(base)

    gpu = GPUVectorIndex()
    gpu.load_cpu_index(base)
    gpu.to_gpu(device_id=0)  # should gracefully fallback when GPU/FAISS GPU unavailable

    results = gpu.search([0.9, 0.1, 0.0, 0.0], 1)
    assert results
    assert results[0][0] == "a"

    gpu.add([0.0, 0.0, 1.0, 0.0], "c")
    results2 = gpu.search([0.0, 0.0, 1.0, 0.0], 1)
    assert results2
    assert results2[0][0] == "c"
