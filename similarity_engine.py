from typing import List, Tuple, Set


class SimilarityEngine:
    def __init__(
        self,
        minhash,
        lsh,
        database,
        threshold: float = 0.75,
        debug: bool = False,
    ):
        """
        minhash  - объект MinHash
        lsh      - объект LSHIndex
        database - объект SQLitePersistence
        threshold - минимальный порог similarity
        debug     - печать служебной информации
        """
        self.minhash = minhash
        self.lsh = lsh
        self.database = database
        self.threshold = threshold
        self.debug = debug

    # -------------------------
    # EXACT JACCARD
    # -------------------------
    def exact_jaccard(self, set1: Set, set2: Set) -> float:
        if not set1 or not set2:
            return 0.0

        intersection = len(set1 & set2)
        union = len(set1 | set2)

        if union == 0:
            return 0.0

        return intersection / union

    # -------------------------
    # MAIN SEARCH (HYBRID)
    # -------------------------
    def search(self, query_ngrams: List[Tuple]) -> List[Tuple[str, float]]:
        """
        Hybrid search:
        1) LSH candidate retrieval
        2) Exact Jaccard scoring
        3) Fallback to full scan if LSH returned nothing
        """

        query_set = set(query_ngrams)

        if not query_set:
            return []

        # 1️⃣ Compute MinHash signature
        query_signature = self.minhash.compute_signature(query_set)

        # 2️⃣ Get LSH candidates
        candidates = self.lsh.query(query_signature)

        if self.debug:
            print(f"[DEBUG] LSH candidates: {len(candidates)}")

        # 3️⃣ Decide search space
        if candidates:
            search_space = candidates
        else:
            # Fallback: full scan
            if self.debug:
                print("[DEBUG] LSH returned no candidates → fallback to full DB scan")
            search_space = self.database.get_all_tracks()

        results = []

        # 4️⃣ Exact similarity scoring
        for track_id in search_space:
            stored_ngrams = self.database.get_ngrams(track_id)

            if not stored_ngrams:
                continue

            score = self.exact_jaccard(query_set, set(stored_ngrams))

            if score >= self.threshold:
                results.append((track_id, score))

        # 5️⃣ Sort descending
        results.sort(key=lambda x: x[1], reverse=True)

        return results