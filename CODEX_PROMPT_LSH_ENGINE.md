# Codex Master Prompt — Build LSH Engine for Mantra

Repository:
https://github.com/Mantra-AI-Hub/mantra-prototype

Project name:
Mantra — AI music similarity engine for detecting melodic plagiarism in MIDI files.

Existing architecture:

MIDI → Pitch Extraction → Interval Normalization → Fingerprint → Similarity Score

Current modules already implemented:

similarity_engine.py
sqlite_persistence.py
mantra.py
tests/
ARCHITECTURE.md
ALGORITHM.md

CI pipeline:
GitHub Actions with pytest.

-----------------------------------------------------

TASK

Implement a scalable Locality Sensitive Hashing (LSH) indexing engine for interval fingerprints.

The goal is to accelerate similarity search across large MIDI datasets.

Target scale:

100k – 1M MIDI files

-----------------------------------------------------

ARCHITECTURE

Create a new module group:

indexing/

Files to implement:

indexing/lsh_index.py
indexing/hash_functions.py
indexing/bucket_store.py

-----------------------------------------------------

DATA FLOW

Interval fingerprint example:

[2, 2, -1, 5, -3, 4, -2]

↓

MinHash / band hashing

↓

LSH tables

↓

Candidate track IDs

↓

Similarity scoring

-----------------------------------------------------

IMPLEMENTATION DETAILS

Hash strategy:

Use MinHash style hashing on interval sequences.

Steps:

1 normalize interval sequence
2 generate k hash functions
3 produce signature vector
4 split signature into bands
5 assign buckets

-----------------------------------------------------

CONFIG PARAMETERS

num_hashes = 128
num_bands = 16
rows_per_band = 8

Allow configuration via constructor.

-----------------------------------------------------

CORE CLASS

class LSHIndex:

Methods:

add(track_id, fingerprint)

index_dataset(list_of_tracks)

query(fingerprint)

get_candidates(signature)

-----------------------------------------------------

BUCKET STORE

Use dictionary-based storage:

band_hash → list of track_ids

Optionally support persistence via SQLite.

-----------------------------------------------------

HASH FUNCTIONS

Implement:

generate_hash_functions()
compute_minhash_signature()

Use deterministic hashing.

-----------------------------------------------------

PERFORMANCE TARGETS

Index:

10k MIDI files < 20 seconds

Query:

< 50 ms

-----------------------------------------------------

TESTING

Add new tests:

tests/test_lsh_index.py

Test cases:

• identical fingerprints
• transposed melodies
• random sequences
• collision control
• large dataset simulation

-----------------------------------------------------

CODE QUALITY

Requirements:

Python 3.11+

Type hints

Docstrings

Modular design

Clear comments

-----------------------------------------------------

INTEGRATION

Integrate with existing search pipeline:

mantra.py

Before computing similarity:

retrieve LSH candidates.

-----------------------------------------------------

EXPECTED RESULT

Fully working LSH indexing engine integrated into Mantra.

-----------------------------------------------------

OUTPUT FORMAT

Generate full Python code for:

hash_functions.py
bucket_store.py
lsh_index.py

Plus:

tests/test_lsh_index.py