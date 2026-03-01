from builder.fingerprint_builder import FingerprintBuilder
from indexing.similarity_engine import SimilarityEngine

builder = FingerprintBuilder()
engine = SimilarityEngine()

# Build fingerprints
fp1 = builder.build_from_midi("test.mid")
fp2 = builder.build_from_midi("test.mid")  # тот же файл

engine.add_fingerprint("track_1", fp1)

results = engine.query_similarity(fp2)

print("=== SIMILARITY RESULTS ===")
for track_id, score in results:
    print(track_id, round(score, 3))