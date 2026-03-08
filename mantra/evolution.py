"""CLI for evolving track genomes via the EvolutionEngine."""

from __future__ import annotations

import argparse

import numpy as np

from mantra.intelligence.evolution_engine import EvolutionEngine


def _summary_line(gen: int, fitnesses: list[float], best_track: str, best_score: float) -> str:
    mean_fitness = float(np.mean(fitnesses)) if fitnesses else 0.0
    return f"Generation {gen}: best={best_track} score={best_score:.3f} mean={mean_fitness:.3f}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Evolutionary track optimization CLI")
    parser.add_argument("--generations", type=int, default=3, help="Number of generations to run")
    parser.add_argument("--population", type=int, default=32, help="Starting population size")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic random seed")
    args = parser.parse_args()

    engine = EvolutionEngine(seed=int(args.seed))
    engine.initialize_population(int(args.population))

    best_track_id = ""
    best_score = 0.0
    print("Evolution Engine Run")
    print("====================")
    for gen in range(1, max(1, int(args.generations)) + 1):
        population, fitnesses = engine.run_generation()
        scores = [f.fitness for f in fitnesses]
        if scores:
            current_best = fitnesses[np.argmax(scores)]
            if current_best.fitness >= best_score:
                best_score = current_best.fitness
                best_track_id = current_best.track_id
        print(_summary_line(gen, scores, best_track_id or "n/a", best_score))
    print("Final best track:", best_track_id or "none")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
