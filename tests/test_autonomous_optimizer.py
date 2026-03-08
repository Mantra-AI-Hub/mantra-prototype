from mantra.autonomous_optimizer import AutonomousOptimizer


def test_autonomous_optimizer_updates_state():
    opt = AutonomousOptimizer()
    opt.monitor_metrics({"ctr": 0.1, "latency": 0.3})
    state = opt.optimize()
    assert state["iterations"] == 1
    assert 0.0 <= state["exploration"] <= 1.0

