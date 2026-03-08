import sys
import os
from mido import MidiFile
from itertools import islice

from mantra.sqlite_persistence import SQLitePersistence
from mantra.similarity_engine import SimilarityEngine


# ============================================================
# CONFIG
# ============================================================

NGRAM_SIZE = 3
NUM_HASHES = 64
NUM_BANDS = 8
THRESHOLD = 0.75


# ============================================================
# SIMPLE MINHASH IMPLEMENTATION
# ============================================================

import random


class MinHash:
    def __init__(self, num_hashes=64):
        self.num_hashes = num_hashes
        self.max_hash = 2**32 - 1
        self.hash_funcs = self._generate_hash_funcs()

    def _generate_hash_funcs(self):
        funcs = []
        for _ in range(self.num_hashes):
            a = random.randint(1, self.max_hash)
            b = random.randint(0, self.max_hash)
            funcs.append((a, b))
        return funcs

    def compute_signature(self, values):
        signature = []

        for a, b in self.hash_funcs:
            min_hash = self.max_hash
            for v in values:
                h = hash(v)
                combined = (a * h + b) % self.max_hash
                if combined < min_hash:
                    min_hash = combined
            signature.append(min_hash)

        return signature


# ============================================================
# SIMPLE LSH
# ============================================================

class LSH:
    def __init__(self, num_bands=8):
        self.num_bands = num_bands
        self.buckets = {}

    def index(self, track_id, signature):
        rows_per_band = len(signature) // self.num_bands

        for band in range(self.num_bands):
            start = band * rows_per_band
            end = start + rows_per_band
            band_tuple = tuple(signature[start:end])

            key = (band, band_tuple)

            if key not in self.buckets:
                self.buckets[key] = []

            self.buckets[key].append(track_id)

    def query(self, signature):
        rows_per_band = len(signature) // self.num_bands
        candidates = set()

        for band in range(self.num_bands):
            start = band * rows_per_band
            end = start + rows_per_band
            band_tuple = tuple(signature[start:end])

            key = (band, band_tuple)

            if key in self.buckets:
                for track_id in self.buckets[key]:
                    candidates.add(track_id)

        return list(candidates)


# ============================================================
# MIDI ANALYSIS
# ============================================================

def extract_intervals(midi_path):
    mid = MidiFile(midi_path)

    notes = []

    for msg in mid.tracks[0]:
        if msg.type == "note_on" and msg.velocity > 0:
            notes.append(msg.note)

    intervals = []

    for i in range(1, len(notes)):
        intervals.append(notes[i] - notes[i - 1])

    return intervals


def build_ngrams(intervals, n=3):
    return [
        tuple(intervals[i:i + n])
        for i in range(len(intervals) - n + 1)
    ]


# ============================================================
# CLI COMMANDS
# ============================================================

def analyze(file_path):
    intervals = extract_intervals(file_path)
    ngrams = build_ngrams(intervals, NGRAM_SIZE)

    print("=== ANALYSIS ===")
    print(f"Source: {file_path}")
    print(f"Intervals: {tuple(intervals)}")
    print(f"N-gram count: {len(ngrams)}")


def add_track(file_path, track_id, db, minhash, lsh):
    intervals = extract_intervals(file_path)
    ngrams = build_ngrams(intervals, NGRAM_SIZE)

    db.add_track(track_id, ngrams)

    signature = minhash.compute_signature(set(ngrams))
    lsh.index(track_id, signature)

    print(f"Track '{track_id}' added to index.")


def add_folder(folder_path, db, minhash, lsh):
    count = 0

    for filename in os.listdir(folder_path):
        if filename.endswith(".mid"):
            path = os.path.join(folder_path, filename)
            track_id = os.path.splitext(filename)[0]

            intervals = extract_intervals(path)
            ngrams = build_ngrams(intervals, NGRAM_SIZE)

            db.add_track(track_id, ngrams)

            signature = minhash.compute_signature(set(ngrams))
            lsh.index(track_id, signature)

            print(f"Indexed: {track_id}")
            count += 1

    print(f"\nTotal indexed: {count}")


def search(file_path, db, minhash, lsh):
    intervals = extract_intervals(file_path)
    query_ngrams = build_ngrams(intervals, NGRAM_SIZE)

    engine = SimilarityEngine(
        minhash=minhash,
        lsh=lsh,
        database=db,
        threshold=THRESHOLD,
        debug=True
    )

    results = engine.search(query_ngrams)

    print("=== SEARCH RESULTS ===")

    if not results:
        print("No similar tracks found.")
        return

    for track_id, score in results:
        print(track_id, round(score, 3))

    if results and results[0][1] >= THRESHOLD:
        print("\n⚠ Potential plagiarism detected")
        print(f"Threshold: {THRESHOLD}")


# ============================================================
# MAIN
# ============================================================

def main():
    if len(sys.argv) < 3:
        print("Usage:")
        print("  analyze <file>")
        print("  add <file> <track_id>")
        print("  add-folder <folder>")
        print("  search <file>")
        return

    command = sys.argv[1]

    db = SQLitePersistence()
    minhash = MinHash(NUM_HASHES)
    lsh = LSH(NUM_BANDS)

    if command == "analyze":
        analyze(sys.argv[2])

    elif command == "add":
        add_track(sys.argv[2], sys.argv[3], db, minhash, lsh)

    elif command == "add-folder":
        add_folder(sys.argv[2], db, minhash, lsh)

    elif command == "search":
        search(sys.argv[2], db, minhash, lsh)

    else:
        print("Unknown command")


if __name__ == "__main__":
    main()