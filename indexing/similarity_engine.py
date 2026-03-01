from typing import Dict, List, Tuple
from fingerprint.creative_fingerprint import CreativeFingerprint

from core_algorithms.minhash import MinHasher, estimate_similarity
from indexing.lsh_index import LSHIndex
from persistence.sqlite_store import SQLiteStore


class SimilarityEngine:
    """
    Scalable similarity engine with:
        MinHash + LSH + SQLite persistence
    """

    def __init__(
        self,
        num_hashes: int = 64,
        num_bands: int = 8,
        db_path: str = "fingerprints.db"
    ):
        self._storage: Dict[str, CreativeFingerprint] = {}
        self._signatures: Dict[str, List[int]] = {}

        self.minhasher = MinHasher(num_hashes=num_hashes)
        self.lsh = LSHIndex(num_bands=num_bands)
        self.store = SQLiteStore(db_path)

        self._load_from_db()

    # ==========================================================
    # LOAD EXISTING
    # ==========================================================

    def _load_from_db(self):
        for fid, signature in self.store.load_all():
            self._signatures[fid] = signature
            self.lsh.add(fid, signature)

    # ==========================================================
    # ADD
    # ==========================================================

    def add_fingerprint(
        self,
        fingerprint_id: str,
        fingerprint: CreativeFingerprint
    ):
        self._storage[fingerprint_id] = fingerprint

        signature = self.minhasher.signature(
            fingerprint.melody.ngram_hashes
        )

        self._signatures[fingerprint_id] = signature
        self.lsh.add(fingerprint_id, signature)

        self.store.save_signature(fingerprint_id, signature)

    # ==========================================================
    # QUERY
    # ==========================================================

    def query_similarity(
        self,
        query_fp: CreativeFingerprint,
        top_k: int = 5
    ) -> List[Tuple[str, float]]:

        query_signature = self.minhasher.signature(
            query_fp.melody.ngram_hashes
        )

        candidate_ids = self.lsh.query(query_signature)

        results: List[Tuple[str, float]] = []

        for fid in candidate_ids:
            target_signature = self._signatures[fid]

            similarity = estimate_similarity(
                query_signature,
                target_signature
            )

            results.append((fid, similarity))

        results.sort(key=lambda x: x[1], reverse=True)

        return results[:top_k]