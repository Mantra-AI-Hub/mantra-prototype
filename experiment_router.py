"""Experiment routing and logging for A/B testing."""

from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional


class ExperimentRouter:
    def __init__(self, db_path: str = "data/experiments.db"):
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
                CREATE TABLE IF NOT EXISTS experiments (
                    experiment_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    group_a_model TEXT NOT NULL,
                    group_b_model TEXT NOT NULL,
                    active INTEGER NOT NULL DEFAULT 1
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS experiment_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id TEXT NOT NULL,
                    result_json TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT (datetime('now'))
                )
                """
            )
            conn.commit()

    def create_experiment(self, experiment_id: str, name: str, model_a: str, model_b: str) -> Dict[str, str]:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO experiments(experiment_id, name, group_a_model, group_b_model, active)
                VALUES (?, ?, ?, ?, 1)
                ON CONFLICT(experiment_id) DO UPDATE SET
                    name=excluded.name,
                    group_a_model=excluded.group_a_model,
                    group_b_model=excluded.group_b_model,
                    active=excluded.active
                """,
                (str(experiment_id), str(name), str(model_a), str(model_b)),
            )
            conn.commit()
        return {
            "experiment_id": str(experiment_id),
            "name": str(name),
            "group_a_model": str(model_a),
            "group_b_model": str(model_b),
        }

    def list_experiments(self) -> List[Dict[str, object]]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM experiments ORDER BY experiment_id").fetchall()
        return [dict(row) for row in rows]

    def _get_active(self) -> Optional[Dict[str, object]]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM experiments WHERE active = 1 ORDER BY experiment_id LIMIT 1").fetchone()
        return dict(row) if row else None

    def route_experiment(self, user_id: str) -> Dict[str, str]:
        experiment = self._get_active()
        if not experiment:
            return {"experiment_id": "default", "group": "A", "model": "default"}

        digest = hashlib.sha1(str(user_id).encode("utf-8")).hexdigest()
        bucket = int(digest[:8], 16) % 2
        if bucket == 0:
            return {
                "experiment_id": str(experiment["experiment_id"]),
                "group": "A",
                "model": str(experiment["group_a_model"]),
            }
        return {
            "experiment_id": str(experiment["experiment_id"]),
            "group": "B",
            "model": str(experiment["group_b_model"]),
        }

    def log_result(self, experiment_id: str, result) -> None:
        import json

        payload = json.dumps(result)
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO experiment_logs(experiment_id, result_json) VALUES (?, ?)",
                (str(experiment_id), payload),
            )
            conn.commit()
