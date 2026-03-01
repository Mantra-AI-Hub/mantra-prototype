from datetime import datetime
from typing import List

from extraction.midi_parser import parse_midi_notes
from core_algorithms.interval import build_interval_sequence
from core_algorithms.rolling_hash import polynomial_hash, ngram_hashes

from fingerprint.metadata import Metadata
from fingerprint.melody_features import MelodyFeatures
from fingerprint.creative_fingerprint import CreativeFingerprint


class FingerprintBuilder:
    """
    V1 Monophonic Fingerprint Builder.

    Pipeline:
        MIDI → Notes → Intervals → Rolling Hash → N-grams → Fingerprint
    """

    def __init__(self, ngram_size: int = 4):
        self.ngram_size = ngram_size

    # ==========================================================
    # PUBLIC API
    # ==========================================================

    def build_from_midi(self, file_path: str) -> CreativeFingerprint:
        """
        Main entry point for building CreativeFingerprint from MIDI.
        """

        # 1️⃣ Extract pitch sequence
        notes: List[int] = parse_midi_notes(file_path)

        # If too short → return empty fingerprint
        if len(notes) < 2:
            return self._build_empty_fingerprint(file_path)

        # 2️⃣ Build normalized interval sequence
        interval_sequence = build_interval_sequence(notes)

        # 3️⃣ Compute stable rolling hash
        interval_hash = polynomial_hash(interval_sequence)

        # 4️⃣ Compute n-gram rolling hashes
        ngram_hash_list = ngram_hashes(
            interval_sequence,
            self.ngram_size
        )

        # 5️⃣ Compute repetition metric
        repetition_score = self._compute_repetition_score(
            ngram_hash_list
        )

        # 6️⃣ Metadata
        metadata = Metadata(
            fingerprint_version="1.0",
            created_at=datetime.utcnow(),
            source_name=file_path
        )

        # 7️⃣ Melody features
        melody = MelodyFeatures(
            interval_sequence=tuple(interval_sequence),
            interval_hash=interval_hash,
            ngram_hashes=tuple(ngram_hash_list),
            repetition_score=repetition_score
        )

        return CreativeFingerprint(
            metadata=metadata,
            melody=melody
        )

    # ==========================================================
    # INTERNAL HELPERS
    # ==========================================================

    def _compute_repetition_score(self, ngram_hashes: List[int]) -> float:
        """
        Measures motif repetition density.

        0.0 → all fragments unique
        1.0 → all fragments identical
        """
        if not ngram_hashes:
            return 0.0

        unique = len(set(ngram_hashes))
        total = len(ngram_hashes)

        return 1.0 - (unique / total)

    def _build_empty_fingerprint(self, file_path: str) -> CreativeFingerprint:
        """
        Handles very short MIDI cases safely.
        """
        metadata = Metadata(
            fingerprint_version="1.0",
            created_at=datetime.utcnow(),
            source_name=file_path
        )

        melody = MelodyFeatures(
            interval_sequence=tuple(),
            interval_hash=0,
            ngram_hashes=tuple(),
            repetition_score=0.0
        )

        return CreativeFingerprint(
            metadata=metadata,
            melody=melody
        )