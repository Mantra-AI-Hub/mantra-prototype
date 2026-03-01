from builder.fingerprint_builder import FingerprintBuilder


def main():
    midi_path = "test.mid"

    builder = FingerprintBuilder(ngram_size=3)

    fingerprint = builder.build_from_midi(midi_path)

    print("\n=== CREATIVE FINGERPRINT V1 ===")
    print("Source:", fingerprint.metadata.source_name)
    print("Version:", fingerprint.metadata.fingerprint_version)

    melody = fingerprint.melody

    print("\n--- Melody Features ---")
    print("Interval sequence:", melody.interval_sequence)
    print("Interval count:", len(melody.interval_sequence))
    print("Interval hash:", melody.interval_hash)
    print("N-gram count:", len(melody.ngram_hashes))
    print("Repetition score:", round(melody.repetition_score, 4))


if __name__ == "__main__":
    main()