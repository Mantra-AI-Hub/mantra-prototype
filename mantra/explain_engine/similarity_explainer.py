"""Human-readable similarity explanations for MIDI pairs."""

from typing import Dict, List, Sequence, Tuple

import numpy as np

from mantra.dataset_engine.dataset_scanner import load_midi
from mantra.dataset_engine.melody_extractor import (
    extract_interval_sequence,
    extract_melody,
    extract_pitch_sequence,
)
from mantra.fingerprinting.midi.fingerprint import build_fingerprint_from_pitch
from mantra.similarity.similarity import pitch_similarity


def _ngram_patterns(values: Sequence[int], n: int = 3) -> List[Tuple[int, ...]]:
    """Return list of n-gram patterns."""
    if len(values) < n:
        return []
    return [tuple(values[i : i + n]) for i in range(len(values) - n + 1)]


def _jaccard_similarity(a: Sequence[int], b: Sequence[int]) -> float:
    """Compute Jaccard similarity between two sequences."""
    set_a = set(a)
    set_b = set(b)

    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0

    return len(set_a & set_b) / len(set_a | set_b)


def _safe_cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity with zero/shape protection."""
    if a.shape != b.shape:
        return 0.0
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _rhythm_similarity(melody_a: List[Tuple[int, float]], melody_b: List[Tuple[int, float]]) -> float:
    """
    Basic rhythm similarity heuristic based on normalized note durations.
    """
    durations_a = np.array([d for _, d in melody_a], dtype=float)
    durations_b = np.array([d for _, d in melody_b], dtype=float)

    if durations_a.size == 0 and durations_b.size == 0:
        return 1.0
    if durations_a.size == 0 or durations_b.size == 0:
        return 0.0

    # Compare the overlapping prefix for stability.
    size = min(durations_a.size, durations_b.size)
    va = durations_a[:size]
    vb = durations_b[:size]

    if va.sum() > 0:
        va = va / va.sum()
    if vb.sum() > 0:
        vb = vb / vb.sum()

    return _safe_cosine(va, vb)


def explain_similarity(trackA: str, trackB: str) -> Dict[str, object]:
    """
    Explain why two MIDI tracks are considered similar.

    Returned fields:
    - similarity_score
    - shared_interval_patterns
    - rhythm_similarity
    - pitch_similarity
    """
    midi_a = load_midi(trackA)
    midi_b = load_midi(trackB)

    pitch_a = extract_pitch_sequence(midi_a)
    pitch_b = extract_pitch_sequence(midi_b)

    intervals_a = extract_interval_sequence(midi_a)
    intervals_b = extract_interval_sequence(midi_b)

    melody_a = extract_melody(midi_a)
    melody_b = extract_melody(midi_b)

    interval_overlap = _jaccard_similarity(intervals_a, intervals_b)

    patterns_a = set(_ngram_patterns(intervals_a, n=3))
    patterns_b = set(_ngram_patterns(intervals_b, n=3))
    shared_patterns = sorted(patterns_a & patterns_b)

    fp_a = build_fingerprint_from_pitch(pitch_a)
    fp_b = build_fingerprint_from_pitch(pitch_b)
    pitch_score = float(pitch_similarity(fp_a.pitch, fp_b.pitch))

    rhythm_score = _rhythm_similarity(melody_a, melody_b)

    total_score = float(np.clip(0.4 * interval_overlap + 0.4 * pitch_score + 0.2 * rhythm_score, 0.0, 1.0))

    return {
        "similarity_score": total_score,
        "shared_interval_patterns": shared_patterns,
        "rhythm_similarity": rhythm_score,
        "pitch_similarity": pitch_score,
        "interval_overlap": interval_overlap,
    }
