"""Scanning and loading utilities for MIDI datasets."""

from pathlib import Path
from typing import List, Union

import mido


PathLike = Union[str, Path]


def scan_dataset(folder: PathLike) -> List[str]:
    """
    Return a sorted list of MIDI file paths inside a dataset folder.

    The scan is recursive and includes `.mid` and `.midi` files.
    """
    root = Path(folder)
    if not root.exists() or not root.is_dir():
        return []

    midi_files = [
        str(path)
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".mid", ".midi"}
    ]
    midi_files.sort()
    return midi_files


def load_midi(file: PathLike) -> mido.MidiFile:
    """Load a MIDI file from disk."""
    return mido.MidiFile(str(file))
