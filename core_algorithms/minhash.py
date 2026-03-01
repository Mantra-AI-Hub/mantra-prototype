import random
from typing import Iterable, List


class MinHasher:
    """
    MinHash signature generator for Jaccard similarity approximation.
    """

    def __init__(self, num_hashes: int = 64, seed: int = 42):
        self.num_hashes = num_hashes
        random.seed(seed)

        # Generate random hash coefficients
        self._hash_params = [
            (random.randint(1, 2**31 - 1), random.randint(0, 2**31 - 1))
            for _ in range(num_hashes)
        ]

        self._prime = 4294967311  # large prime > 2^32

    # ==========================================================
    # PUBLIC API
    # ==========================================================

    def signature(self, values: Iterable[int]) -> List[int]:
        """
        Generate MinHash signature for a set of integers.
        """

        values_set = set(values)

        if not values_set:
            return [0] * self.num_hashes

        signature = []

        for a, b in self._hash_params:
            min_hash = float("inf")

            for value in values_set:
                hash_value = (a * value + b) % self._prime
                if hash_value < min_hash:
                    min_hash = hash_value

            signature.append(int(min_hash))

        return signature


# ==========================================================
# Utility
# ==========================================================

def estimate_similarity(sig_a: List[int], sig_b: List[int]) -> float:
    """
    Estimates Jaccard similarity from two MinHash signatures.
    """

    if len(sig_a) != len(sig_b):
        raise ValueError("Signature lengths must match")

    matches = sum(1 for a, b in zip(sig_a, sig_b) if a == b)
    return matches / len(sig_a)