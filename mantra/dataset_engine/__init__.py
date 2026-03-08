"""Dataset ingestion utilities for MIDI corpora."""

from mantra.dataset_engine.dataset_scanner import load_midi, scan_dataset
from mantra.dataset_engine.melody_extractor import (
    extract_interval_sequence,
    extract_melody,
    extract_pitch_sequence,
)

__all__ = [
    "scan_dataset",
    "load_midi",
    "extract_melody",
    "extract_pitch_sequence",
    "extract_interval_sequence",
]
