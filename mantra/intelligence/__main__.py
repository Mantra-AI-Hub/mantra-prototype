"""CLI entry point for the Global Music Intelligence engine."""

from __future__ import annotations

import argparse

from mantra.intelligence.global_music_intelligence import GlobalMusicIntelligenceEngine


def _format_patterns(patterns: list, label: str) -> str:
    lines = [f"{label} ({len(patterns)}):"]
    for pattern in patterns:
        lines.append(
            f"  - {pattern['feature_name']}: score={pattern['correlation_score']:.3f} impact={pattern['impact_strength']:.3f}"
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Global music intelligence diagnostics")
    parser.add_argument("--report", action="store_true", help="Generate trend report")
    parser.add_argument("--top-k", type=int, default=3, help="Number of features to highlight")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for deterministic behavior")
    args = parser.parse_args()
    if not args.report:
        parser.print_help()
        return 1
    engine = GlobalMusicIntelligenceEngine(seed=args.seed)
    report = engine.generate_trend_report(top_k=max(1, args.top_k))
    pos_patterns = report.get("top_positive_features", [])
    neg_patterns = report.get("top_negative_features", [])
    print("Global Music Intelligence Report")
    print("=" * 32)
    print(_format_patterns(pos_patterns, "Top Positive Features"))
    print()
    print(_format_patterns(neg_patterns, "Top Negative Features"))
    print()
    print("Sampled Engagement Metrics:")
    for entry in report.get("sampled_engagement", []):
        print(f"  - {entry['track_id']}: virality={entry['virality_score']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
