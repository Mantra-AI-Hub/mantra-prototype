from dataclasses import dataclass
from typing import List
import numpy as np

from mantra.core_algorithms.interval import compute_intervals, normalize_intervals


@dataclass
class Fingerprint:
    pitch: np.ndarray
    rhythm: np.ndarray
    contour: List[int]


def build_fingerprint_from_pitch(notes: List[int]) -> Fingerprint:
    """
    Build simplified fingerprint from raw MIDI pitch sequence.
    """

    if not notes:
        return Fingerprint(
            pitch=np.zeros(12),
            rhythm=np.zeros(1),
            contour=[]
        )

    # ---- Pitch Class Histogram (12-dim) ----
    pitch_classes = [n % 12 for n in notes]
    pitch_vector = np.zeros(12)

    for pc in pitch_classes:
        pitch_vector[pc] += 1

    pitch_vector = pitch_vector / np.linalg.norm(pitch_vector) if np.linalg.norm(pitch_vector) else pitch_vector

    # ---- Rhythm (simplified as length-based placeholder) ----
    rhythm_vector = np.array([len(notes)], dtype=float)

    # ---- Contour (interval direction) ----
    intervals = compute_intervals(notes)
    normalized = normalize_intervals(intervals)

    contour = normalized

    return Fingerprint(
        pitch=pitch_vector,
        rhythm=rhythm_vector,
        contour=contour
    )
