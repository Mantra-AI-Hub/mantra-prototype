"""Melody extraction helpers for MIDI datasets."""

from typing import List, Tuple

import mido

from mantra.core_algorithms.interval import build_interval_sequence
from mantra.dataset_engine.dataset_scanner import load_midi


MelodyNote = Tuple[int, float]


def _extract_track_melody(track: mido.MidiTrack, ticks_per_beat: int) -> List[MelodyNote]:
    """
    Extract monophonic note events from one MIDI track.

    Returns a list of (pitch, duration_beats).
    """
    melody: List[MelodyNote] = []
    active_notes: dict[int, int] = {}
    absolute_tick = 0

    for msg in track:
        absolute_tick += msg.time

        if msg.type == "note_on" and msg.velocity > 0:
            active_notes[msg.note] = absolute_tick
            continue

        if msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
            start_tick = active_notes.pop(msg.note, None)
            if start_tick is None:
                continue

            duration_ticks = max(0, absolute_tick - start_tick)
            duration_beats = duration_ticks / ticks_per_beat if ticks_per_beat else 0.0
            melody.append((msg.note, duration_beats))

    return melody


def extract_melody(midi: mido.MidiFile) -> List[MelodyNote]:
    """
    Extract a representative melody track from MIDI.

    Selection heuristic:
    - ignore likely drum tracks when possible
    - choose the track with the largest number of note events
    """
    best: List[MelodyNote] = []

    for track in midi.tracks:
        notes = _extract_track_melody(track, midi.ticks_per_beat)
        if len(notes) > len(best):
            best = notes

    return best


def extract_pitch_sequence(midi: mido.MidiFile) -> List[int]:
    """Convert extracted melody to ordered pitch sequence."""
    return [pitch for pitch, _ in extract_melody(midi)]


def extract_interval_sequence(midi: mido.MidiFile) -> List[int]:
    """
    Convert extracted melody to normalized interval sequence.

    Reuses the existing interval pipeline for consistency with fingerprints.
    """
    pitches = extract_pitch_sequence(midi)
    return build_interval_sequence(pitches)


def extract_pitch_sequence_from_file(path: str) -> List[int]:
    """Convenience helper for direct file input."""
    return extract_pitch_sequence(load_midi(path))


def extract_interval_sequence_from_file(path: str) -> List[int]:
    """Convenience helper for direct file input."""
    return extract_interval_sequence(load_midi(path))
