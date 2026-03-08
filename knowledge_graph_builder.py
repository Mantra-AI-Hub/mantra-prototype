"""Global music knowledge graph builder."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List


class KnowledgeGraphBuilder:
    def __init__(self):
        self.nodes: Dict[str, Dict[str, object]] = {}
        self.edges: List[Dict[str, object]] = []

    def add_node(self, node_id: str, node_type: str, attrs: Dict[str, object] | None = None) -> None:
        self.nodes[str(node_id)] = {"type": str(node_type), **(attrs or {})}

    def add_edge(self, src: str, dst: str, relation: str, weight: float = 1.0) -> None:
        self.edges.append({"src": str(src), "dst": str(dst), "relation": str(relation), "weight": float(weight)})

    def build_from_records(self, records: Iterable[Dict[str, object]]) -> Dict[str, object]:
        for rec in records:
            user = str(rec.get("user_id") or "")
            track = str(rec.get("track_id") or "")
            artist = str(rec.get("artist") or "")
            genre = str(rec.get("genre") or "")
            playlist = str(rec.get("playlist_id") or "")
            if user:
                self.add_node(user, "user")
            if track:
                self.add_node(track, "track")
            if artist:
                self.add_node(artist, "artist")
            if genre:
                self.add_node(genre, "genre")
            if playlist:
                self.add_node(playlist, "playlist")

            if user and track:
                self.add_edge(user, track, "listens")
            if user and artist:
                self.add_edge(user, artist, "likes")
            if track and artist:
                self.add_edge(track, artist, "created_by")
            if track and genre:
                self.add_edge(track, genre, "belongs_to")
            if playlist and track:
                self.add_edge(playlist, track, "contains")
        return self.graph()

    def graph(self) -> Dict[str, object]:
        return {"nodes": self.nodes, "edges": self.edges}

