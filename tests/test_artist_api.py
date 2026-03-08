from mantra.interfaces.api.api_server import CreateArtistRequest, ai_artist_create


def test_artist_creation_endpoint():
    payload = CreateArtistRequest(name="Nebula", genre="ambient", persona="calm")
    response = ai_artist_create(payload)
    assert response["status"] == "created"
    assert "artist" in response
    assert response["artist"]["name"] == "Nebula"
