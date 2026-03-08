"""Virtual artist creation and discography simulation."""

from __future__ import annotations

import logging
import uuid
from typing import Dict, List


class VirtualArtistFactory:
    def __init__(self) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self.artists: Dict[str, Dict[str, object]] = {}
        self.metrics: Dict[str, float | int] = {"artists_created": 0, "tracks_generated": 0}

    def create_ai_artist(self, name: str, genre: str) -> Dict[str, object]:
        artist_id = f"va_{uuid.uuid4().hex[:8]}"
        artist = {"artist_id": artist_id, "name": str(name), "genre": str(genre), "persona": {}, "discography": []}
        self.artists[artist_id] = artist
        self.metrics["artists_created"] = int(self.metrics["artists_created"]) + 1
        return artist

    def assign_persona(self, artist_id: str, persona: Dict[str, object]) -> Dict[str, object]:
        if artist_id not in self.artists:
            raise ValueError(f"Unknown artist_id: {artist_id}")
        self.artists[artist_id]["persona"] = dict(persona)
        return dict(self.artists[artist_id])

    def generate_discography(self, artist_id: str, tracks: int = 5) -> List[Dict[str, object]]:
        if artist_id not in self.artists:
            raise ValueError(f"Unknown artist_id: {artist_id}")
        created = []
        for i in range(max(1, int(tracks))):
            created.append({"track_id": f"{artist_id}_track_{i}", "title": f"Track {i}", "artist_id": artist_id})
        self.artists[artist_id]["discography"] = created
        self.metrics["tracks_generated"] = int(self.metrics["tracks_generated"]) + len(created)
        self.logger.info("Generated discography for %s with %d tracks", artist_id, len(created))
        return created

    def list_artists(self) -> List[Dict[str, object]]:
        return [dict(value) for value in self.artists.values()]

    def metrics_snapshot(self) -> Dict[str, float | int]:
        return dict(self.metrics)
