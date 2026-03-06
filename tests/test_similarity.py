import pytest

from mantra.fingerprinting.midi.fingerprint import build_fingerprint_from_pitch
from mantra.similarity.similarity import calculate_similarity


def test_identical_similarity():
    notes = [60, 62, 64, 65, 67]

    f1 = build_fingerprint_from_pitch(notes)
    f2 = build_fingerprint_from_pitch(notes)

    result = calculate_similarity(f1, f2)

    assert round(result, 4) == 1.0


def test_random_similarity():
    a = [60, 61, 62]
    b = [70, 71, 72]

    f1 = build_fingerprint_from_pitch(a)
    f2 = build_fingerprint_from_pitch(b)

    result = calculate_similarity(f1, f2)

    assert 0.0 <= result <= 1.0
