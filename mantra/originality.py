"""CLI entry point for MANTRA track originality checks."""

from __future__ import annotations

import argparse

import numpy as np

from mantra.intelligence.global_music_intelligence import GlobalMusicIntelligenceEngine


def _seed_from_track_id(track_id: str) -> int:
    encoded = np.frombuffer(track_id.encode("utf-8"), dtype=np.uint8)
    if encoded.size == 0:
        return 0
    weights = np.arange(1, encoded.size + 1, dtype=np.uint64)
    return int(np.dot(encoded.astype(np.uint64), weights) % np.uint64(2**32 - 1))


def main() -> int:
    parser = argparse.ArgumentParser(description="MANTRA track originality CLI")
    parser.add_argument("--track", required=True, help="Track identifier to evaluate")
    parser.add_argument("--seed", type=int, default=0, help="Seed used for the reference library")
    args = parser.parse_args()

    engine = GlobalMusicIntelligenceEngine(seed=int(args.seed))
    engine.ingest_experiment_data(population_size=12, generations=2)
    originality = engine.assess_originality({"track_id": args.track}, seed=_seed_from_track_id(args.track))

    print("Track originality report")
    print(f"Similarity: {originality.similarity_score:.2f}")
    print(f"Originality: {originality.originality_score:.2f}")
    if originality.most_similar_track:
        print(f"Most similar track: {originality.most_similar_track}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
