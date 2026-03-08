"""Conversational music assistant prompt handling."""

from __future__ import annotations

import re
from typing import Dict


def handle_query(prompt: str) -> Dict[str, object]:
    text = str(prompt or "").strip()
    lower = text.lower()

    if "playlist" in lower:
        match = re.search(r"track[_\s-]*([a-zA-Z0-9]+)", lower)
        seed = f"track_{match.group(1)}" if match else None
        return {
            "intent": "playlist_generate",
            "seed_track_id": seed,
            "length": 10,
            "message": "Generating playlist from seed track." if seed else "Generating playlist from recommendations.",
        }

    if any(word in lower for word in ["lyrics", "lyric"]):
        return {
            "intent": "search_lyrics",
            "query": text,
            "message": "Searching by lyrics.",
        }

    if any(word in lower for word in ["find", "search", "similar"]):
        return {
            "intent": "search_text",
            "query": text,
            "message": "Searching music catalog by text query.",
        }

    return {
        "intent": "help",
        "message": "Try asking for similar tracks, lyrics search, or playlist generation.",
    }
