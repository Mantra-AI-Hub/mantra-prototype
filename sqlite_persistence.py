import sqlite3
import json


class SQLitePersistence:
    def __init__(self, db_path="fingerprints.db"):
        self.conn = sqlite3.connect(db_path)
        self._create_table()

    def _create_table(self):
        cursor = self.conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS fingerprints (
                track_id TEXT,
                ngram TEXT
            )
        """)
        self.conn.commit()

    # ---------------------
    # SAVE
    # ---------------------
    def add_track(self, track_id, ngrams):
        cursor = self.conn.cursor()

        for ng in ngrams:
            cursor.execute(
                "INSERT INTO fingerprints VALUES (?, ?)",
                (track_id, json.dumps(ng))
            )

        self.conn.commit()

    # ---------------------
    # LOAD
    # ---------------------
    def get_ngrams(self, track_id):
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT ngram FROM fingerprints WHERE track_id = ?",
            (track_id,)
        )
        rows = cursor.fetchall()
        return [tuple(json.loads(row[0])) for row in rows]

    def get_all_tracks(self):
        cursor = self.conn.cursor()
        cursor.execute("SELECT DISTINCT track_id FROM fingerprints")
        rows = cursor.fetchall()
        return [row[0] for row in rows]

    def clear(self):
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM fingerprints")
        self.conn.commit()