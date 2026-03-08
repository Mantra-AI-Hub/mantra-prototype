"""MANTRA system diagnostic CLI."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
from typing import Callable, Dict, List


CheckResult = Dict[str, str]


def _ok(name: str, details: str) -> CheckResult:
    return {"name": name, "status": "OK", "details": details}


def _fail(name: str, details: str) -> CheckResult:
    return {"name": name, "status": "FAIL", "details": details}


def check_imports() -> CheckResult:
    name = "Imports"
    modules = [
        "mantra.fingerprint_engine",
        "mantra.vector_index",
        "mantra.vector_index_service",
        "mantra.recommendation_engine",
        "mantra.realtime_recommender",
        "mantra.embedding_trainer",
        "mantra.feature_store",
        "mantra.model_registry",
        "mantra.training_pipeline",
        "mantra.ai_music_supervisor",
        "mantra.self_evolving_recommender",
    ]
    loaded = 0
    try:
        for module in modules:
            importlib.import_module(module)
            loaded += 1
        return _ok(name, f"loaded {loaded}/{len(modules)} critical modules")
    except Exception as exc:
        return _fail(name, str(exc))


def check_vector_index() -> CheckResult:
    name = "Vector Index"
    try:
        from mantra.vector_index import VectorIndex

        index = VectorIndex(dimension=4)
        index.add([1.0, 0.0, 0.0, 0.0], "doctor_probe")
        return _ok(name, f"initialized dimension={index.dimension}")
    except Exception as exc:
        return _fail(name, str(exc))


def check_recommender() -> CheckResult:
    name = "Recommendation Engine"
    try:
        from mantra.recommendation_engine import RecommendationEngine

        class _DummyTrackStore:
            def list_tracks(self):
                return []

            def get_track(self, _track_id: str):
                return None

        engine = RecommendationEngine(track_store=_DummyTrackStore())
        return _ok(name, f"instance={engine.__class__.__name__}")
    except Exception as exc:
        return _fail(name, str(exc))


def check_generation() -> CheckResult:
    name = "AI Generation"
    try:
        from mantra.ai_producer_engine import AIProducerEngine

        engine = AIProducerEngine()
        melody = engine.generate_melody(seed=60, bars=1)
        return _ok(name, f"melody_length={len(melody)}")
    except Exception as exc:
        return _fail(name, str(exc))


def check_feature_store() -> CheckResult:
    name = "Feature Store"
    try:
        from mantra.feature_store import FeatureStore

        store = FeatureStore(db_path="data/doctor_feature_store.db")
        store.store_user_features("doctor_user", {"health": "ok"})
        value = store.get_user_features("doctor_user") or {}
        return _ok(name, f"user_features_keys={len(value.keys())}")
    except Exception as exc:
        return _fail(name, str(exc))


def check_models() -> CheckResult:
    name = "Model Registry"
    try:
        from mantra.model_registry import ModelRegistry

        registry = ModelRegistry(root_dir="data/doctor_models")
        registry.register("doctor_model", "v0", "data/doctor_models/doctor_model.bin")
        latest = registry.get_latest("doctor_model") or {}
        return _ok(name, f"latest={latest.get('version', 'none')}")
    except Exception as exc:
        return _fail(name, str(exc))


def check_supervisor() -> CheckResult:
    name = "AI Supervisor"
    try:
        from mantra.ai_music_supervisor import AIMusicSupervisor

        status = AIMusicSupervisor().status()
        return _ok(name, f"keys={','.join(sorted(status.keys()))}")
    except Exception as exc:
        return _fail(name, str(exc))


def check_experiments() -> CheckResult:
    name = "Experiments"
    try:
        from mantra.autonomous_experimentation import AutonomousExperimentation

        exp = AutonomousExperimentation()
        launched = exp.launch_ab_experiment("a", "b")
        return _ok(name, f"id={launched['experiment_id']}")
    except Exception as exc:
        return _fail(name, str(exc))


def check_self_evolving() -> CheckResult:
    name = "Self-Evolving Recommender"
    try:
        from mantra.self_evolving_recommender import SelfEvolvingRecommender

        model = SelfEvolvingRecommender()
        model.monitor_model_performance(0.8)
        return _ok(name, f"strategy={model.metrics_snapshot().get('current_strategy')}")
    except Exception as exc:
        return _fail(name, str(exc))


def check_api() -> CheckResult:
    name = "API Server"
    try:
        spec = importlib.util.find_spec("mantra.interfaces.api.api_server")
        if spec is None:
            return _fail(name, "module spec not found")
        return _ok(name, "module spec resolved")
    except Exception as exc:
        return _fail(name, str(exc))


def run_checks() -> List[CheckResult]:
    checks: List[Callable[[], CheckResult]] = [
        check_imports,
        check_vector_index,
        check_recommender,
        check_generation,
        check_feature_store,
        check_models,
        check_supervisor,
        check_experiments,
        check_self_evolving,
        check_api,
    ]
    return [check() for check in checks]


def metrics_report() -> Dict[str, object]:
    report: Dict[str, object] = {}
    try:
        from mantra.self_evolving_recommender import SelfEvolvingRecommender

        report["self_evolving_recommender"] = SelfEvolvingRecommender().metrics_snapshot()
    except Exception as exc:
        report["self_evolving_recommender"] = {"error": str(exc)}
    try:
        from mantra.meta_learning_engine import MetaLearningEngine

        report["meta_learning_engine"] = MetaLearningEngine(store_path="data/doctor_meta_learning.json").metrics_snapshot()
    except Exception as exc:
        report["meta_learning_engine"] = {"error": str(exc)}
    try:
        from mantra.autonomous_experimentation import AutonomousExperimentation

        report["experiments"] = AutonomousExperimentation().metrics_snapshot()
    except Exception as exc:
        report["experiments"] = {"error": str(exc)}
    try:
        from mantra.auto_scaling_ai import AutoScalingAI

        report["autoscaling"] = AutoScalingAI().metrics_snapshot()
    except Exception as exc:
        report["autoscaling"] = {"error": str(exc)}
    try:
        from mantra.distributed_training_orchestrator import DistributedTrainingOrchestrator

        report["distributed_training"] = DistributedTrainingOrchestrator().status().get("metrics", {})
    except Exception as exc:
        report["distributed_training"] = {"error": str(exc)}
    return report


def _render_human(results: List[CheckResult]) -> str:
    lines = ["MANTRA SYSTEM DIAGNOSTIC", "========================", ""]
    for item in results:
        label = str(item["name"])
        dots = "." * max(1, 24 - len(label))
        lines.append(f"{label} {dots} {item['status']}")
    lines.append("")
    overall = "HEALTHY" if all(item["status"] == "OK" for item in results) else "ISSUES DETECTED"
    lines.append(f"Overall Status: {overall}")
    return "\n".join(lines)


def build_report() -> Dict[str, object]:
    results = run_checks()
    overall = "HEALTHY" if all(item["status"] == "OK" for item in results) else "ISSUES DETECTED"
    return {"overall_status": overall, "checks": results}


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="MANTRA diagnostic tool")
    parser.add_argument("--json", action="store_true", dest="as_json", help="Output report as JSON")
    parser.add_argument("--metrics", action="store_true", help="Output module metrics")
    args = parser.parse_args(argv)

    if args.metrics:
        metrics = metrics_report()
        if args.as_json:
            print(json.dumps({"metrics": metrics}, indent=2))
        else:
            print("MANTRA DIAGNOSTIC METRICS")
            print("=========================")
            print(json.dumps(metrics, indent=2))
        return 0

    report = build_report()
    if args.as_json:
        print(json.dumps(report, indent=2))
    else:
        print(_render_human(report["checks"]))
    return 0 if report["overall_status"] == "HEALTHY" else 1


if __name__ == "__main__":
    raise SystemExit(main())
