from mantra.autonomous_experimentation import AutonomousExperimentation


def test_autonomous_experimentation_flow():
    exp = AutonomousExperimentation()
    launched = exp.launch_ab_experiment("a", "b")
    evaluated = exp.evaluate_experiment(launched["experiment_id"], [0.6, 0.7], [0.5, 0.6])
    assert evaluated["winner"] == "A"
    deployed = exp.deploy_best_performer(launched["experiment_id"])
    assert deployed == "a"
