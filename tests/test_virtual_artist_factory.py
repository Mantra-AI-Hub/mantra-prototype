from mantra.virtual_artist_factory import VirtualArtistFactory


def test_virtual_artist_factory_flow():
    factory = VirtualArtistFactory()
    artist = factory.create_ai_artist("Nova", "electronic")
    updated = factory.assign_persona(artist["artist_id"], {"mood": "futuristic"})
    tracks = factory.generate_discography(artist["artist_id"], tracks=3)
    assert updated["persona"]["mood"] == "futuristic"
    assert len(tracks) == 3
