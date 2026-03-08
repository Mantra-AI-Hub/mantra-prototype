"""Evolutionary optimization over synthetic engagement metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from mantra.intelligence.synthetic_listener_engine import SyntheticListenerEngine, TrackEngagement


@dataclass
class TrackGenome:
    track_id: str
    tempo: float
    energy: float
    mood: float
    novelty: float
    rhythm_complexity: float
    harmonic_density: float


@dataclass
class TrackFitness:
    track_id: str
    fitness: float
    virality: float
    completion_rate: float
    replay_rate: float
    skip_rate: float


class EvolutionEngine:
    def __init__(self, seed: int = 24) -> None:
        self._rng = np.random.default_rng(seed)
        self.synthetic = SyntheticListenerEngine(seed=seed)
        self.population: List[TrackGenome] = []
        self.fitnesses: List[TrackFitness] = []
        self.generation: int = 0

    def _random_feature(self) -> float:
        return float(self._rng.uniform(0.1, 0.9))

    def initialize_population(self, population_size: int) -> List[TrackGenome]:
        self.population = []
        n = max(1, int(population_size))
        for idx in range(n):
            genome = TrackGenome(
                track_id=f"track_{self.generation}_{idx}",
                tempo=float(self._rng.uniform(60.0, 180.0)),
                energy=self._random_feature(),
                mood=self._random_feature(),
                novelty=self._random_feature(),
                rhythm_complexity=self._random_feature(),
                harmonic_density=self._random_feature(),
            )
            self.population.append(genome)
        return list(self.population)

    def _genome_to_track(self, genome: TrackGenome) -> Dict[str, object]:
        return {
            "track_id": genome.track_id,
            "vector": [
                genome.tempo / 180.0,
                genome.energy,
                genome.mood,
                genome.novelty,
                genome.rhythm_complexity,
                genome.harmonic_density,
                float((genome.energy + genome.mood) / 2.0),
                float((genome.novelty + genome.rhythm_complexity) / 2.0),
            ],
            "quality_score": (genome.energy + genome.mood) / 2.0,
            "emotional_score": genome.harmonic_density,
            "novelty_score": genome.novelty,
        }

    def mutate_genome(self, genome: TrackGenome) -> TrackGenome:
        delta = lambda value: float(np.clip(value + self._rng.normal(0, 0.05), 0.0, 1.0))
        return TrackGenome(
            track_id=f"{genome.track_id}_m",
            tempo=float(np.clip(genome.tempo + self._rng.normal(0, 5.0), 60.0, 180.0)),
            energy=delta(genome.energy),
            mood=delta(genome.mood),
            novelty=delta(genome.novelty),
            rhythm_complexity=delta(genome.rhythm_complexity),
            harmonic_density=delta(genome.harmonic_density),
        )

    def crossover_genomes(self, parent_a: TrackGenome, parent_b: TrackGenome) -> TrackGenome:
        blend = lambda a, b: float(0.5 * (a + b))
        return TrackGenome(
            track_id=f"{parent_a.track_id}_{parent_b.track_id}_c",
            tempo=blend(parent_a.tempo, parent_b.tempo),
            energy=blend(parent_a.energy, parent_b.energy),
            mood=blend(parent_a.mood, parent_b.mood),
            novelty=blend(parent_a.novelty, parent_b.novelty),
            rhythm_complexity=blend(parent_a.rhythm_complexity, parent_b.rhythm_complexity),
            harmonic_density=blend(parent_a.harmonic_density, parent_b.harmonic_density),
        )

    def compute_fitness(self, metrics: TrackEngagement) -> TrackFitness:
        if metrics.plays == 0:
            completion_rate = 0.0
            replay_rate = 0.0
            skip_rate = 0.0
        else:
            completion_rate = metrics.completions / metrics.plays
            replay_rate = metrics.replays / metrics.plays
            skip_rate = metrics.skips / metrics.plays
        score = metrics.virality_score + completion_rate * 0.4 + replay_rate * 0.3 - skip_rate * 0.5
        fitness = float(np.clip(score, 0.0, 1.0))
        return TrackFitness(
            track_id=metrics.track_id,
            fitness=fitness,
            virality=metrics.virality_score,
            completion_rate=completion_rate,
            replay_rate=replay_rate,
            skip_rate=skip_rate,
        )

    def select_top_tracks(self, population: List[Tuple[TrackGenome, TrackFitness]], top_k: int) -> List[Tuple[TrackGenome, TrackFitness]]:
        sorted_pop = sorted(population, key=lambda pair: pair[1].fitness, reverse=True)
        return sorted_pop[: max(1, int(top_k))]

    def evolve_population(self) -> List[TrackGenome]:
        pool = []
        track_pool = [self._genome_to_track(g) for g in self.population]
        for genome in self.population:
            engagement = self.synthetic.simulate_release(genome.track_id, track_pool)
            fitness = self.compute_fitness(engagement)
            pool.append((genome, fitness))
        elite = self.select_top_tracks(pool, top_k=max(1, len(self.population) // 4))
        offspring: List[TrackGenome] = []
        for idx in range(len(self.population)):
            parent_a = elite[idx % len(elite)][0]
            parent_b = elite[(idx + 1) % len(elite)][0]
            child = self.crossover_genomes(parent_a, parent_b)
            offspring.append(self.mutate_genome(child))
        self.fitnesses = [fitness for _, fitness in pool]
        self.population = offspring
        self.generation += 1
        return list(self.population)

    def run_generation(self) -> Tuple[List[TrackGenome], List[TrackFitness]]:
        population = self.evolve_population()
        return population, list(self.fitnesses)
