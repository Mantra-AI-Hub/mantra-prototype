from typing import List


OCTAVE = 12
HALF_OCTAVE = 6


def compute_intervals(notes: List[int]) -> List[int]:
    """
    Computes raw pitch intervals between consecutive MIDI notes.

    Example:
        [60, 64, 67] -> [4, 3]
    """
    if len(notes) < 2:
        return []

    intervals: List[int] = []

    for i in range(1, len(notes)):
        intervals.append(notes[i] - notes[i - 1])

    return intervals


def normalize_intervals(intervals: List[int]) -> List[int]:
    """
    Normalizes intervals into minimal pitch-class range [-6, 6].

    Logic:
        1. Reduce to mod 12
        2. If value > 6 → invert downward (value -= 12)

    Examples:
        7  -> -5
        8  -> -4
        -7 -> 5
        12 -> 0
    """
    normalized: List[int] = []

    for interval in intervals:
        value = interval % OCTAVE

        if value > HALF_OCTAVE:
            value -= OCTAVE

        normalized.append(value)

    return normalized


def build_interval_sequence(notes: List[int]) -> List[int]:
    """
    Full V1 interval pipeline.

    Steps:
        1. Compute raw intervals
        2. Normalize to minimal representation [-6, 6]

    Returns:
        List[int] — normalized interval sequence
    """
    raw_intervals = compute_intervals(notes)
    normalized_intervals = normalize_intervals(raw_intervals)

    return normalized_intervals