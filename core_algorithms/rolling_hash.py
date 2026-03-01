from typing import List


BASE = 31
MOD = 1_000_000_007


def polynomial_hash(sequence: List[int]) -> int:
    """
    Deterministic polynomial rolling hash.

    H = sum(ai * BASE^(n-i-1)) mod MOD
    """
    h = 0

    for value in sequence:
        # смещаем интервалы в положительный диапазон
        adjusted = value + 12  # интервалы [-6,6] → [6,18]

        h = (h * BASE + adjusted) % MOD

    return h


def ngram_hashes(sequence: List[int], n: int) -> List[int]:
    """
    Computes rolling hashes for all n-grams.
    """
    if len(sequence) < n:
        return []

    hashes: List[int] = []

    for i in range(len(sequence) - n + 1):
        fragment = sequence[i:i + n]
        hashes.append(polynomial_hash(fragment))

    return hashes