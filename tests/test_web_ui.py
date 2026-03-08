import mantra.interfaces.api.api_server as api_server


def test_web_ui_routes_available():
    response = api_server.web_index()
    assert response.path.endswith("web\\index.html") or response.path.endswith("web/index.html")
