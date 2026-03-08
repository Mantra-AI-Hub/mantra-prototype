from mantra.ecosystem_simulator import EcosystemSimulator


def test_ecosystem_simulator_runs():
    sim = EcosystemSimulator(seed=1)
    state = sim.simulate_users_artists_playlists(n_users=10, n_artists=5, n_playlists=3)
    assert len(state["users"]) == 10
    trends = sim.simulate_trend_emergence()
    assert "trend_score" in trends
