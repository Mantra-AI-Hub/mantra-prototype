"""Persistent SQLite track metadata and fingerprint storage."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from mantra.database.schema import CREATE_INDEXES_SQL, FINGERPRINTS_TABLE_SQL, TRACKS_TABLE_SQL


TrackMetadata = Dict[str, object]
Fingerprint = List[Tuple[str, int]]


class TrackStore:
    """SQLite-backed storage for indexed track metadata and fingerprints."""

    def __init__(self, db_path: str = "mantra_tracks.db"):
        self.db_path = str(db_path)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _initialize(self) -> None:
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(TRACKS_TABLE_SQL)
            connection.execute(FINGERPRINTS_TABLE_SQL)
            self._ensure_track_columns(connection)
            for statement in CREATE_INDEXES_SQL:
                connection.execute(statement)
            connection.commit()

    @staticmethod
    def _ensure_track_columns(connection: sqlite3.Connection) -> None:
        existing = {
            str(row["name"])
            for row in connection.execute("PRAGMA table_info(tracks)").fetchall()
        }
        required = {
            "artist": "TEXT",
            "album": "TEXT",
            "genre": "TEXT",
            "tags": "TEXT",
            "year": "INTEGER",
        }
        for name, sql_type in required.items():
            if name not in existing:
                connection.execute(f"ALTER TABLE tracks ADD COLUMN {name} {sql_type}")

    def add_track(self, metadata: TrackMetadata) -> None:
        """Insert or update track metadata."""
        created_at = metadata.get("created_at")
        if not created_at:
            created_at = datetime.now(timezone.utc).isoformat()

        tags = metadata.get("tags")
        if isinstance(tags, list):
            serialized_tags = json.dumps([str(tag) for tag in tags])
        elif isinstance(tags, str) and tags.strip():
            serialized_tags = json.dumps([tag.strip() for tag in tags.split(",") if tag.strip()])
        else:
            serialized_tags = json.dumps([])

        payload = {
            "track_id": str(metadata["track_id"]),
            "filename": str(metadata["filename"]),
            "duration": float(metadata["duration"]),
            "embedding_path": str(metadata["embedding_path"]),
            "fingerprint_hash_count": int(metadata["fingerprint_hash_count"]),
            "created_at": str(created_at),
            "artist": str(metadata.get("artist") or ""),
            "album": str(metadata.get("album") or ""),
            "genre": str(metadata.get("genre") or ""),
            "tags": serialized_tags,
            "year": int(metadata["year"]) if metadata.get("year") is not None else None,
        }

        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO tracks (
                    track_id, filename, duration, embedding_path, fingerprint_hash_count, created_at,
                    artist, album, genre, tags, year
                )
                VALUES (
                    :track_id, :filename, :duration, :embedding_path, :fingerprint_hash_count, :created_at,
                    :artist, :album, :genre, :tags, :year
                )
                ON CONFLICT(track_id) DO UPDATE SET
                    filename=excluded.filename,
                    duration=excluded.duration,
                    embedding_path=excluded.embedding_path,
                    fingerprint_hash_count=excluded.fingerprint_hash_count,
                    created_at=excluded.created_at,
                    artist=excluded.artist,
                    album=excluded.album,
                    genre=excluded.genre,
                    tags=excluded.tags,
                    year=excluded.year
                """,
                payload,
            )
            connection.commit()

    @staticmethod
    def _normalize_track_row(row: sqlite3.Row | None) -> Optional[TrackMetadata]:
        if row is None:
            return None
        data = dict(row)
        raw_tags = data.get("tags")
        if isinstance(raw_tags, str):
            try:
                parsed = json.loads(raw_tags)
                if isinstance(parsed, list):
                    data["tags"] = [str(tag) for tag in parsed]
                else:
                    data["tags"] = []
            except json.JSONDecodeError:
                data["tags"] = []
        elif isinstance(raw_tags, list):
            data["tags"] = [str(tag) for tag in raw_tags]
        else:
            data["tags"] = []
        return data

    def get_track(self, track_id: str) -> Optional[TrackMetadata]:
        """Return a track metadata record by ID."""
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM tracks WHERE track_id = ?", (str(track_id),)).fetchone()
        return self._normalize_track_row(row)

    def list_tracks(self) -> List[TrackMetadata]:
        """List all track metadata records."""
        with self._connect() as connection:
            rows = connection.execute("SELECT * FROM tracks ORDER BY created_at DESC").fetchall()
        return [self._normalize_track_row(row) for row in rows if row is not None]

    def delete_track(self, track_id: str) -> None:
        """Delete a track and associated fingerprint data."""
        with self._connect() as connection:
            connection.execute("DELETE FROM tracks WHERE track_id = ?", (str(track_id),))
            connection.commit()

    def upsert_fingerprint(self, track_id: str, fingerprint: Sequence[Tuple[str, int]]) -> None:
        """Persist fingerprint entries for a track."""
        serialized = json.dumps([[str(hash_key), int(time_bin)] for hash_key, time_bin in fingerprint])
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO fingerprints (track_id, fingerprint_json)
                VALUES (?, ?)
                ON CONFLICT(track_id) DO UPDATE SET
                    fingerprint_json=excluded.fingerprint_json
                """,
                (str(track_id), serialized),
            )
            connection.commit()

    def get_fingerprint(self, track_id: str) -> Fingerprint:
        """Load fingerprint entries for a track."""
        with self._connect() as connection:
            row = connection.execute(
                "SELECT fingerprint_json FROM fingerprints WHERE track_id = ?", (str(track_id),)
            ).fetchone()
        if not row:
            return []
        parsed = json.loads(str(row["fingerprint_json"]))
        return [(str(item[0]), int(item[1])) for item in parsed]

    def list_fingerprints(self) -> Dict[str, Fingerprint]:
        """Load all stored fingerprints keyed by track ID."""
        with self._connect() as connection:
            rows = connection.execute("SELECT track_id, fingerprint_json FROM fingerprints").fetchall()

        data: Dict[str, Fingerprint] = {}
        for row in rows:
            parsed = json.loads(str(row["fingerprint_json"]))
            data[str(row["track_id"])] = [(str(item[0]), int(item[1])) for item in parsed]
        return data


_DEFAULT_STORE = TrackStore()


def add_track(metadata: TrackMetadata) -> None:
    _DEFAULT_STORE.add_track(metadata)


def get_track(track_id: str) -> Optional[TrackMetadata]:
    return _DEFAULT_STORE.get_track(track_id)


def list_tracks() -> List[TrackMetadata]:
    return _DEFAULT_STORE.list_tracks()


def delete_track(track_id: str) -> None:
    _DEFAULT_STORE.delete_track(track_id)
