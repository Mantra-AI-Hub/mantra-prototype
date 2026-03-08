"""Command-line interface for indexing, search, and similarity explanation."""

import argparse
import json
from pathlib import Path
from typing import Iterable, Optional

from mantra.dataset_engine.dataset_scanner import scan_dataset
from mantra.explain_engine.similarity_explainer import explain_similarity
from mantra.fingerprinting.midi.fingerprint_builder import FingerprintBuilder
from mantra.index.similarity_engine import SimilarityEngine


def _build_track_id(dataset_root: Path, midi_path: str) -> str:
    """Build a stable track id relative to the indexed dataset root."""
    path = Path(midi_path)
    try:
        return str(path.relative_to(dataset_root)).replace("\\", "/")
    except ValueError:
        return path.name


def _cmd_index(args: argparse.Namespace) -> int:
    midi_files = scan_dataset(args.folder)
    if not midi_files:
        print("No MIDI files found.")
        return 1

    engine = SimilarityEngine(
        num_hashes=args.num_hashes,
        num_bands=args.num_bands,
        db_path=args.db_path,
    )
    builder = FingerprintBuilder(ngram_size=args.ngram_size)
    root = Path(args.folder).resolve()

    for midi_path in midi_files:
        fingerprint = builder.build_from_midi(midi_path)
        track_id = _build_track_id(root, midi_path)
        engine.add_fingerprint(track_id, fingerprint)

    print(f"Indexed {len(midi_files)} MIDI files into {args.db_path}")
    return 0


def _cmd_search(args: argparse.Namespace) -> int:
    builder = FingerprintBuilder(ngram_size=args.ngram_size)
    engine = SimilarityEngine(
        num_hashes=args.num_hashes,
        num_bands=args.num_bands,
        db_path=args.db_path,
    )

    query_fp = builder.build_from_midi(args.midi)
    results = engine.query_similarity(query_fp, top_k=args.top_k)

    print("Search results:")
    if not results:
        print("No similar tracks found.")
        return 0

    for track_id, score in results:
        print(f"{track_id}\t{score:.4f}")
    return 0


def _cmd_explain(args: argparse.Namespace) -> int:
    explanation = explain_similarity(args.midi_a, args.midi_b)
    print(json.dumps(explanation, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser for Mantra commands."""
    parser = argparse.ArgumentParser(prog="mantra", description="MANTRA music intelligence CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    index_parser = subparsers.add_parser("index", help="Index a folder of MIDI files")
    index_parser.add_argument("folder", help="Dataset folder containing MIDI files")
    index_parser.add_argument("--db-path", default="fingerprints.db", help="SQLite database path")
    index_parser.add_argument("--ngram-size", type=int, default=4, help="N-gram size for melody fingerprinting")
    index_parser.add_argument("--num-hashes", type=int, default=64, help="Number of MinHash functions")
    index_parser.add_argument("--num-bands", type=int, default=8, help="Number of LSH bands")
    index_parser.set_defaults(handler=_cmd_index)

    search_parser = subparsers.add_parser("search", help="Search similar tracks for a MIDI file")
    search_parser.add_argument("midi", help="Query MIDI file path")
    search_parser.add_argument("--db-path", default="fingerprints.db", help="SQLite database path")
    search_parser.add_argument("--top-k", type=int, default=5, help="Number of matches to return")
    search_parser.add_argument("--ngram-size", type=int, default=4, help="N-gram size for melody fingerprinting")
    search_parser.add_argument("--num-hashes", type=int, default=64, help="Number of MinHash functions")
    search_parser.add_argument("--num-bands", type=int, default=8, help="Number of LSH bands")
    search_parser.set_defaults(handler=_cmd_search)

    explain_parser = subparsers.add_parser("explain", help="Explain similarity between two MIDI files")
    explain_parser.add_argument("midi_a", help="First MIDI file")
    explain_parser.add_argument("midi_b", help="Second MIDI file")
    explain_parser.set_defaults(handler=_cmd_explain)

    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    """CLI entrypoint."""
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
