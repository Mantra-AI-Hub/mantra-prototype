"""
MIDI → MelodicFingerprint parser.

Supports:
1) Protobuf fingerprint export (legacy mode)
2) Lightweight pitch extraction for V1 FingerprintBuilder
"""

from pathlib import Path
from typing import List, Tuple

import mido

from mantra.protos import fingerprint_pb2


class MIDIParser:
    """
    Parses MIDI files and extracts monophonic note sequences.
    """

    def __init__(self):
        pass

    # ==========================================================
    # PROTOBUF MODE (existing system)
    # ==========================================================

    def parse_file(self, midi_path: str) -> fingerprint_pb2.MelodicFingerprint:
        """
        Reads MIDI and returns a protobuf MelodicFingerprint.
        """
        midi = mido.MidiFile(midi_path)

        notes = self._extract_monophonic_notes(midi)

        fingerprint = fingerprint_pb2.MelodicFingerprint()
        fingerprint.source_file = Path(midi_path).name

        if len(notes) < 2:
            return fingerprint

        # compute intervals
        for i in range(1, len(notes)):
            prev_pitch, prev_duration = notes[i - 1]
            curr_pitch, _ = notes[i]

            interval = fingerprint.intervals.add()
            interval.value = curr_pitch - prev_pitch
            interval.duration = prev_duration

        return fingerprint

    def save_to_file(
        self,
        fingerprint: fingerprint_pb2.MelodicFingerprint,
        output_path: str,
    ) -> None:
        with open(output_path, "wb") as f:
            f.write(fingerprint.SerializeToString())

    # ==========================================================
    # INTERNAL NOTE EXTRACTION
    # ==========================================================

    def _extract_monophonic_notes(
        self, midi: mido.MidiFile
    ) -> List[Tuple[int, float]]:
        """
        Returns list of (pitch, duration_seconds).

        Assumes monophonic melodic line.
        """
        notes: List[Tuple[int, float]] = []
        active_notes = {}
        current_time = 0

        tempo = 500000  # default 120 BPM
        ticks_per_beat = midi.ticks_per_beat
        seconds_per_tick = (tempo / 1_000_000) / ticks_per_beat

        for track in midi.tracks:
            current_time = 0

            for msg in track:
                current_time += msg.time

                if msg.type == "set_tempo":
                    tempo = msg.tempo
                    seconds_per_tick = (tempo / 1_000_000) / ticks_per_beat

                if msg.type == "note_on" and msg.velocity > 0:
                    active_notes[msg.note] = current_time

                elif msg.type == "note_off" or (
                    msg.type == "note_on" and msg.velocity == 0
                ):
                    if msg.note in active_notes:
                        start_tick = active_notes.pop(msg.note)
                        duration_ticks = current_time - start_tick
                        duration_sec = duration_ticks * seconds_per_tick

                        notes.append((msg.note, duration_sec))

        return notes


# ==============================================================
# LIGHTWEIGHT MODE FOR FINGERPRINT BUILDER V1
# ==============================================================

def parse_midi_notes(file_path: str) -> List[int]:
    """
    Lightweight helper for V1 FingerprintBuilder.

    Returns:
        List[int] — ordered MIDI pitches (monophonic)

    Does NOT use protobuf.
    """
    parser = MIDIParser()
    midi = mido.MidiFile(file_path)

    notes = parser._extract_monophonic_notes(midi)

    return [pitch for pitch, _ in notes]


# ==============================================================
# CLI MODE
# ==============================================================

def extract_fingerprint(midi_path: str, output_path: str):
    parser = MIDIParser()
    fp = parser.parse_file(midi_path)
    parser.save_to_file(fp, output_path)

    print(f"✅ Saved fingerprint: {output_path}")
    print(f"Intervals: {len(fp.intervals)}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print("Usage: python midi_parser.py input.mid output.bin")
    else:
        extract_fingerprint(sys.argv[1], sys.argv[2])
