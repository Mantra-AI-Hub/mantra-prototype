from collections import defaultdict
from protos import fingerprint_pb2


def load_intervals(path: str):
    fp = fingerprint_pb2.MelodicFingerprint()
    with open(path, "rb") as f:
        fp.ParseFromString(f.read())
    return list(fp.intervals)


def generate_ngrams(sequence, n=3):
    return [tuple(sequence[i:i+n]) for i in range(len(sequence) - n + 1)]


class HashIndex:
    def __init__(self, n=3):
        self.n = n
        self.index = defaultdict(set)  # hash -> {song_ids}
        self.song_data = {}  # song_id -> intervals

    def _hash_ngram(self, ngram):
        return hash(ngram)

    def add_song(self, song_id: str, fingerprint_path: str):
        intervals = load_intervals(fingerprint_path)
        self.song_data[song_id] = intervals

        ngrams = generate_ngrams(intervals, self.n)

        for ng in ngrams:
            h = self._hash_ngram(ng)
            self.index[h].add(song_id)

    def search(self, fingerprint_path
