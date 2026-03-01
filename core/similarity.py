# core/similarity.py

import numpy as np
from dataclasses import dataclass


# =========================================================
# Weights (can be tuned later, but must stay deterministic)
# =========================================================

W_PITCH = 0.45
W_RHYTHM = 0.30
W_CONTOUR = 0.25


# =========================================================
# Fingerprint structure
# =========================================================

@dataclass
class Fingerprint:
    pitch: np.ndarray
    rhythm: np.ndarray
    contour: list  # list of tuples (n-grams)


# =========================================================
# Internal math utilities
# =========================================================

def _safe_cosine(a: np.ndarray, b: np.ndarray) -> float:
    """
    Cosine similarity with shape validation and zero protection.
    """

    if a.shape != b.shape:
        raise ValueError("Vectors must have identical shape")

    denom = np.linalg.norm(a) * np.linalg.norm(b)

    if denom == 0:
        return 0.0

    return float(np.dot(a, b) / denom)


# =========================================================
# Component similarities
# =========================================================

def pitch_similarity(p1: np.ndarray, p2: np.ndarray) -> float:
    """
    Transposition-invariant pitch similarity.
    Checks 12 cyclic shifts.
    """

    if p1.shape != p2.shape:
        raise ValueError("Pitch vectors must have identical shape")

    max_sim = 0.0

    for shift in range(12):
        shifted = np.roll(p2, shift)
        sim = _safe_cosine(p1, shifted)
        max_sim = max(max_sim, sim)

    return max_sim


def rhythm_similarity(r1: np.ndarray, r2: np.ndarray) -> float:
    """
    Direct cosine similarity for rhythm vectors.
    """
    return _safe_cosine(r1, r2)


def contour_similarity(c1, c2) -> float:
    """
    Jaccard similarity over contour n-grams.
    Accepts list of tuples.
    """

    set1 = set(c1)
    set2 = set(c2)

    if not set1 and not set2:
        return 1.0

    union = set1 | set2
    intersection = set1 & set2

    if not union:
        return 0.0

    return len(intersection) / len(union)


# =========================================================
# Public API
# =========================================================

def calculate_similarity(f1: Fingerprint, f2: Fingerprint) -> float:
    """
    Main deterministic similarity function.
    This is the ONLY public entry point.
    """

    s_pitch = pitch_similarity(f1.pitch, f2.pitch)
    s_rhythm = rhythm_similarity(f1.rhythm, f2.rhythm)
    s_contour = contour_similarity(f1.contour, f2.contour)

    total = (
        W_PITCH * s_pitch +
        W_RHYTHM * s_rhythm +
        W_CONTOUR * s_contour
    )

    # Clamp for numerical stability
    return float(np.clip(total, 0.0, 1.0))