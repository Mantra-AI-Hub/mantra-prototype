"""Real-time stream recognition using rolling audio buffer."""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from mantra.fingerprint_engine import generate_fingerprints_from_audio
from mantra.fingerprint_index import FingerprintIndex


class StreamRecognizer:
    def __init__(self, fingerprint_index: FingerprintIndex, sample_rate: int = 22050, max_seconds: float = 8.0):
        self.fingerprint_index = fingerprint_index
        self.sample_rate = int(sample_rate)
        self.max_samples = int(max_seconds * self.sample_rate)
        self._buffer = np.zeros(0, dtype=np.float32)

    def push_chunk(self, samples: List[float]) -> List[Dict[str, object]]:
        chunk = np.asarray(samples, dtype=np.float32).reshape(-1)
        self._buffer = np.concatenate([self._buffer, chunk])
        if self._buffer.size > self.max_samples:
            self._buffer = self._buffer[-self.max_samples :]
        fingerprints = generate_fingerprints_from_audio((self._buffer, self.sample_rate))
        return self.fingerprint_index.query(fingerprints)
