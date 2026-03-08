"""Ecosystem-facing virtual artist factory wrapper."""

from __future__ import annotations

from typing import Dict, Optional

from mantra.virtual_artist_factory import VirtualArtistFactory as BaseVirtualArtistFactory


class VirtualArtistFactory(BaseVirtualArtistFactory):
    def create_artist(self, name: Optional[str] = None, genre: Optional[str] = None, persona: Optional[str] = None) -> Dict[str, object]:
        artist = super().create_ai_artist(name=str(name or "unnamed"), genre=str(genre or "eclectic"))
        persona_payload = {"persona": persona} if persona else {}
        super().assign_persona(artist["artist_id"], persona_payload)
        super().generate_discography(artist["artist_id"], tracks=3)
        return artist


__all__ = ["VirtualArtistFactory"]
