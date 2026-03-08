from mantra.autoscaler import AutoScaler


def test_autoscaler_scale_up_down_and_hold():
    scaler = AutoScaler(min_workers=1, max_workers=4)

    up = scaler.monitor_metrics({"queue_depth": 30, "ingestion_rate": 5, "worker_failures": 0})
    assert up["action"] in {"scale_up", "hold"}

    down = scaler.monitor_metrics({"queue_depth": 0, "ingestion_rate": 0, "worker_failures": 0})
    assert down["action"] in {"scale_down", "hold"}

    fail = scaler.monitor_metrics({"queue_depth": 5, "ingestion_rate": 5, "worker_failures": 1})
    assert fail["action"] == "scale_down"
