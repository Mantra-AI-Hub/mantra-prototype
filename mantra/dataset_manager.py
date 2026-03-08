"""Dataset registration and ingestion state management."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Dict, List, Optional


class DatasetManager:
    def __init__(self, db_path: str = "data/mantra_tracks.db"):
        self.db_path = db_path
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _initialize(self) -> None:
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS datasets (
                    dataset_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    track_count INTEGER NOT NULL,
                    ingestion_status TEXT NOT NULL
                )
                """
            )
            conn.commit()

    def register_dataset(self, dataset_id: str, name: str) -> Dict[str, object]:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO datasets (dataset_id, name, track_count, ingestion_status)
                VALUES (?, ?, 0, 'registered')
                ON CONFLICT(dataset_id) DO UPDATE SET name=excluded.name
                """,
                (dataset_id, name),
            )
            conn.commit()
        return self.get_dataset(dataset_id) or {}

    def set_status(self, dataset_id: str, status: str, track_count: Optional[int] = None) -> None:
        with self._connect() as conn:
            if track_count is None:
                conn.execute("UPDATE datasets SET ingestion_status = ? WHERE dataset_id = ?", (status, dataset_id))
            else:
                conn.execute(
                    "UPDATE datasets SET ingestion_status = ?, track_count = ? WHERE dataset_id = ?",
                    (status, int(track_count), dataset_id),
                )
            conn.commit()

    def get_dataset(self, dataset_id: str) -> Optional[Dict[str, object]]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM datasets WHERE dataset_id = ?", (dataset_id,)).fetchone()
        return dict(row) if row else None

    def list_datasets(self) -> List[Dict[str, object]]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM datasets ORDER BY dataset_id").fetchall()
        return [dict(row) for row in rows]
