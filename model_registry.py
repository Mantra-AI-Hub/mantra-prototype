"""Versioned model registry for production model lifecycle."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Dict, Optional


class ModelRegistry:
    def __init__(self, root_dir: str = "models"):
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.root_dir / "registry.db"
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _initialize(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS model_versions (
                    model_name TEXT NOT NULL,
                    version TEXT NOT NULL,
                    path TEXT NOT NULL,
                    is_active INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL DEFAULT (datetime('now')),
                    PRIMARY KEY (model_name, version)
                )
                """
            )
            conn.commit()

    def register(self, model_name: str, version: str, path: str) -> Dict[str, str]:
        model_dir = self.root_dir / model_name / version
        model_dir.mkdir(parents=True, exist_ok=True)

        resolved_path = str(Path(path))
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO model_versions(model_name, version, path, is_active)
                VALUES (?, ?, ?, 0)
                ON CONFLICT(model_name, version) DO UPDATE SET path=excluded.path
                """,
                (str(model_name), str(version), resolved_path),
            )
            conn.commit()
        return {"model_name": str(model_name), "version": str(version), "path": resolved_path}

    def get_latest(self, model_name: str) -> Optional[Dict[str, str]]:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT model_name, version, path
                FROM model_versions
                WHERE model_name = ?
                ORDER BY created_at DESC, version DESC
                LIMIT 1
                """,
                (str(model_name),),
            ).fetchone()
        return dict(row) if row else None

    def set_active(self, model_name: str, version: str) -> None:
        with self._connect() as conn:
            conn.execute("UPDATE model_versions SET is_active = 0 WHERE model_name = ?", (str(model_name),))
            conn.execute(
                "UPDATE model_versions SET is_active = 1 WHERE model_name = ? AND version = ?",
                (str(model_name), str(version)),
            )
            conn.commit()

    def get_active(self, model_name: str) -> Optional[Dict[str, str]]:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT model_name, version, path
                FROM model_versions
                WHERE model_name = ? AND is_active = 1
                LIMIT 1
                """,
                (str(model_name),),
            ).fetchone()
        return dict(row) if row else None
