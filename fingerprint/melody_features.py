from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class MelodyFeatures:
    """
    Melodic fingerprint features (monophonic V1).
    """

    # Sequence of pitch intervals (e.g. +2, -1, +5)
    interval_sequence: Tuple[int, ...]

    # Rolling hash of full interval sequence
    interval_hash: int

    # N-gram hashes (motivic fragments)
    ngram_hashes: Tuple[int, ...]

    # Repetition metric (motif density score 0.0 - 1.0)
    repetition_score: float