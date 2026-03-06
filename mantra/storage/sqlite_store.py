import sqlite3
import json
from typing import List, Tuple


class SQLiteStore:
    """
    SQLite persistence for MinHash signatures.
    """

    def __init__(self, db_path: str = "fingerprints.db"):
        self.db_path = db_path
        self._initialize()

    # ==========================================================
    # INIT
    # ==========================================================

    def _initialize(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS fingerprints (
                    id TEXT PRIMARY KEY,
                    signature TEXT
                )
            """)

            conn.commit()

    # ==========================================================
    # SAVE
    # ==========================================================

    def save_signature(self, fingerprint_id: str, signature: List[int]):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute("""
                INSERT OR REPLACE INTO fingerprints (id, signature)
                VALUES (?, ?)
            """, (fingerprint_id, json.dumps(signature)))

            conn.commit()

    # ==========================================================
    # LOAD
    # ==========================================================

    def load_all(self) -> List[Tuple[str, List[int]]]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute("SELECT id, signature FROM fingerprints")

            rows = cursor.fetchall()

        results = []
        for fid, sig_json in rows:
            signature = json.loads(sig_json)
            results.append((fid, signature))

        return results