from mantra.intelligence.evolution_engine import EvolutionEngine


def test_evolution_engine_population_and_determinism():
    engine = EvolutionEngine(seed=10)
    pop_a = engine.initialize_population(6)
    eng_a, _ = engine.run_generation()
    engine_b = EvolutionEngine(seed=10)
    pop_b = engine_b.initialize_population(6)
    eng_b, _ = engine_b.run_generation()
    assert [g.tempo for g in pop_a] == [g.tempo for g in pop_b]
    fitnesses_a = [f.fitness for f in engine.fitnesses]
    fitnesses_b = [f.fitness for f in engine_b.fitnesses]
    assert fitnesses_a == fitnesses_b


def test_evolution_engine_fitness_range():
    engine = EvolutionEngine(seed=5)
    engine.initialize_population(4)
    engine.run_generation()
    assert all(0.0 <= f.fitness <= 1.0 for f in engine.fitnesses)
