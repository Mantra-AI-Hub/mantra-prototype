from pathlib import Path

from mantra.dataset_engine.dataset_scanner import load_midi, scan_dataset
from mantra.dataset_engine.melody_extractor import (
    extract_interval_sequence,
    extract_melody,
    extract_pitch_sequence,
)

def test_scan_dataset_finds_midi_files():
    files = scan_dataset("test_midis")
    assert files
    assert any(path.endswith(".mid") for path in files)


def test_melody_pitch_and_interval_extraction():
    midi_path = Path("test_midis") / "test.mid"
    midi = load_midi(midi_path)
    melody = extract_melody(midi)
    pitches = extract_pitch_sequence(midi)
    intervals = extract_interval_sequence(midi)

    assert melody
    assert len(pitches) >= 2
    assert len(intervals) == len(pitches) - 1
