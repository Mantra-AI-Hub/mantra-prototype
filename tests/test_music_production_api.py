from mantra.interfaces.api.api_server import ProduceMusicRequest, ai_music_produce


def test_music_production_endpoint():
    payload = ProduceMusicRequest(genre="ambient", mood="calm", tempo=110)
    response = ai_music_produce(payload)
    assert response["status"] == "generated"
    assert "track" in response
    assert "melody" in response["track"]
