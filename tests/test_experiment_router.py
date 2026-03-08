from uuid import uuid4

from mantra.experiment_router import ExperimentRouter


def test_experiment_router_route_and_log():
    db_path = f"test_experiment_router_{uuid4().hex}.db"
    router = ExperimentRouter(db_path=db_path)

    router.create_experiment("exp1", "test", "model_a", "model_b")
    route = router.route_experiment("user-123")

    assert route["experiment_id"] == "exp1"
    assert route["group"] in {"A", "B"}

    router.log_result("exp1", {"clicked": True})
    listing = router.list_experiments()
    assert listing
    assert listing[0]["experiment_id"] == "exp1"
