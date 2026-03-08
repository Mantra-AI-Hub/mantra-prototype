"""SQLite schema definitions for MANTRA persistent storage."""

TRACKS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS tracks (
    track_id TEXT PRIMARY KEY,
    filename TEXT NOT NULL,
    duration REAL NOT NULL,
    embedding_path TEXT NOT NULL,
    fingerprint_hash_count INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    artist TEXT,
    album TEXT,
    genre TEXT,
    tags TEXT,
    year INTEGER
);
"""

FINGERPRINTS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS fingerprints (
    track_id TEXT PRIMARY KEY,
    fingerprint_json TEXT NOT NULL,
    FOREIGN KEY(track_id) REFERENCES tracks(track_id) ON DELETE CASCADE
);
"""

CREATE_INDEXES_SQL = [
    "CREATE INDEX IF NOT EXISTS idx_tracks_created_at ON tracks(created_at);",
]
