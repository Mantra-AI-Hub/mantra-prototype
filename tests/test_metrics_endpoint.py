import mantra.interfaces.api.api_server as api_server


def test_metrics_endpoint_prometheus_format():
    response = api_server.metrics_endpoint()
    text = response.body.decode("utf-8")

    assert "mantra_ingestion_rate" in text
    assert "mantra_search_latency" in text
    assert "mantra_cache_hit_rate" in text
