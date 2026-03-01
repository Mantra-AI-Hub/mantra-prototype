from typing import Dict, List, Set, Tuple
from collections import defaultdict


class LSHIndex:
    """
    Locality Sensitive Hashing for MinHash signatures.
    """

    def __init__(self, num_bands: int = 8):
        self.num_bands = num_bands
        self._buckets: List[Dict[int, Set[str]]] = []
        self._signatures: Dict[str, List[int]] = {}

    # ==========================================================

    def add(self, fingerprint_id: str, signature: List[int]):
        """
        Adds MinHash signature to LSH index.
        """

        if not self._buckets:
            self._initialize_buckets(len(signature))

        self._signatures[fingerprint_id] = signature

        band_size = len(signature) // self.num_bands

        for band_idx in range(self.num_bands):
            start = band_idx * band_size
            end = start + band_size

            band = tuple(signature[start:end])
            bucket_hash = hash(band)

            self._buckets[band_idx][bucket_hash].add(fingerprint_id)

    # ==========================================================

    def query(self, signature: List[int]) -> Set[str]:
        """
        Returns candidate fingerprint IDs.
        """

        candidates: Set[str] = set()

        band_size = len(signature) // self.num_bands

        for band_idx in range(self.num_bands):
            start = band_idx * band_size
            end = start + band_size

            band = tuple(signature[start:end])
            bucket_hash = hash(band)

            bucket = self._buckets[band_idx].get(bucket_hash)
            if bucket:
                candidates.update(bucket)

        return candidates

    # ==========================================================

    def _initialize_buckets(self, signature_length: int):
        self._buckets = [
            defaultdict(set) for _ in range(self.num_bands)
        ]