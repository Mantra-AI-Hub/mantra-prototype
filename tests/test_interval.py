import pytest
from mantra.core_algorithms.interval import compute_intervals, normalize_intervals


class TestComputeIntervals:

    def test_basic_intervals(self):
        notes = [60, 64, 67]
        assert compute_intervals(notes) == [4, 3]

    def test_descending(self):
        notes = [67, 64, 60]
        assert compute_intervals(notes) == [-3, -4]

    def test_single_note(self):
        assert compute_intervals([60]) == []

    def test_empty(self):
        assert compute_intervals([]) == []


class TestNormalizeIntervals:

    def test_normalize_within_octave(self):
        intervals = [2, 14, -13, 24, -24]
        expected = [2, 2, -1, 0, 0]
        assert normalize_intervals(intervals) == expected

    def test_normalize_mixed_intervals(self):
        intervals = [7, 19, -5, -17, 12]
        # В minimal-модели 7 → -5
        expected = [-5, -5, -5, -5, 0]
        assert normalize_intervals(intervals) == expected

    def test_edge_cases(self):
        intervals = [6, -6, 7, -7, 8, -8]
        expected = [6, 6, -5, 5, -4, 4]
        assert normalize_intervals(intervals) == expected

    def test_zero(self):
        assert normalize_intervals([0]) == [0]

    def test_empty(self):
        assert normalize_intervals([]) == []
