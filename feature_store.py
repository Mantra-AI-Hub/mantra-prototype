"""Persistent feature store for training and ranking features."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Dict, Iterable, List, Optional


class FeatureStore:
    def __init__(self, db_path: str = "data/feature_store.db"):
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
                CREATE TABLE IF NOT EXISTS features (
                    track_id TEXT PRIMARY KEY,
                    feature_json TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS user_features (
                    user_id TEXT PRIMARY KEY,
                    feature_json TEXT NOT NULL
                )
                """
            )
            conn.commit()

    def put(self, track_id: str, features: Dict[str, object]) -> None:
        payload = json.dumps(features)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO features (track_id, feature_json)
                VALUES (?, ?)
                ON CONFLICT(track_id) DO UPDATE SET feature_json=excluded.feature_json
                """,
                (str(track_id), payload),
            )
            conn.commit()

    def get(self, track_id: str) -> Optional[Dict[str, object]]:
        with self._connect() as conn:
            row = conn.execute("SELECT feature_json FROM features WHERE track_id = ?", (str(track_id),)).fetchone()
        if row is None:
            return None
        return json.loads(str(row["feature_json"]))

    def batch_get(self, track_ids: Iterable[str]) -> List[Dict[str, object] | None]:
        return [self.get(track_id) for track_id in track_ids]

    def all_items(self) -> List[tuple[str, Dict[str, object]]]:
        with self._connect() as conn:
            rows = conn.execute("SELECT track_id, feature_json FROM features").fetchall()
        return [(str(r["track_id"]), json.loads(str(r["feature_json"]))) for r in rows]

    def store_user_features(self, user_id: str, features: Dict[str, object]) -> None:
        payload = json.dumps(features)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO user_features (user_id, feature_json)
                VALUES (?, ?)
                ON CONFLICT(user_id) DO UPDATE SET feature_json=excluded.feature_json
                """,
                (str(user_id), payload),
            )
            conn.commit()

    def get_user_features(self, user_id: str) -> Optional[Dict[str, object]]:
        with self._connect() as conn:
            row = conn.execute("SELECT feature_json FROM user_features WHERE user_id = ?", (str(user_id),)).fetchone()
        if row is None:
            return None
        return json.loads(str(row["feature_json"]))

    def store_track_features(self, track_id: str, features: Dict[str, object]) -> None:
        self.put(track_id, features)

    def get_track_features(self, track_id: str) -> Optional[Dict[str, object]]:
        return self.get(track_id)
