"""Online learning interaction recorder and updater."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Dict


class OnlineLearning:
    def __init__(self, db_path: str = "data/online_learning.db"):
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
                CREATE TABLE IF NOT EXISTS interactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    track_id TEXT NOT NULL,
                    event TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT (datetime('now'))
                )
                """
            )
            conn.commit()

    def record_interaction(self, user_id: str, track_id: str, event: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO interactions(user_id, track_id, event) VALUES (?, ?, ?)",
                (str(user_id), str(track_id), str(event)),
            )
            conn.commit()

    def update_models(self) -> Dict[str, int]:
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) AS c FROM interactions").fetchone()
            total = int(row["c"]) if row else 0
            clicks = conn.execute("SELECT COUNT(*) AS c FROM interactions WHERE event = 'click'").fetchone()
            likes = conn.execute("SELECT COUNT(*) AS c FROM interactions WHERE event = 'like'").fetchone()

        return {
            "total_interactions": int(total),
            "click_events": int(clicks["c"]) if clicks else 0,
            "like_events": int(likes["c"]) if likes else 0,
        }

    def list_interactions(self):
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT user_id, track_id, event, created_at FROM interactions ORDER BY id"
            ).fetchall()
        return [dict(row) for row in rows]
