from protos import fingerprint_pb2


def load_fingerprint(path: str):
    fp = fingerprint_pb2.MelodicFingerprint()
    with open(path, "rb") as f:
        fp.ParseFromString(f.read())
    return list(fp.intervals)


def similarity_score(seq1, seq2):
    """
    Простое сравнение по наибольшей общей подпоследовательности
    (LCS-like sliding window версия).
    """

    if not seq1 or not seq2:
        return 0.0

    max_match = 0

    # Sliding window
    for i in range(len(seq1)):
        for j in range(len(seq2)):
            match = 0
            while (
                i + match < len(seq1)
                and j + match < len(seq2)
                and seq1[i + match] == seq2[j + match]
            ):
                match += 1

            max_match = max(max_match, match)

    return max_match / max(len(seq1), len(seq2))


def compare_fingerprints(path1: str, path2: str):
    seq1 = load_fingerprint(path1)
    seq2 = load_fingerprint(path2)

    score = similarity_score(seq1, seq2)

    print(f"Similarity score: {score:.3f}")
    return score
