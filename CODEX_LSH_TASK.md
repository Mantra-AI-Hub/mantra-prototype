# Codex Task: Implement LSH Index Engine for Mantra

Repository:
https://github.com/Mantra-AI-Hub/mantra-prototype

Project:
Mantra — AI-powered music similarity engine for detecting melodic plagiarism in MIDI files.

Existing modules:

similarity_engine.py
sqlite_persistence.py
mantra.py
tests/

Architecture documentation:

ARCHITECTURE.md
ALGORITHM.md

CI pipeline:
GitHub Actions (pytest)

-------------------------------------

Goal

Implement a Locality Sensitive Hashing (LSH) engine to accelerate similarity search across large MIDI datasets.

-------------------------------------

Requirements

The LSH engine must:

• accept interval fingerprints
• create multiple hash tables
• support fast candidate retrieval
• integrate with existing similarity_engine
• scale to datasets of 100k+ MIDI files

-------------------------------------

Directory structure

Create new modules:

indexing/

indexing/lsh_index.py
indexing/hash_functions.py
indexing/bucket_store.py

-------------------------------------

Functionality

1. Hash interval fingerprints

Convert interval sequences into hash signatures.

Example:

[2, 2, -1, 5, -3]

↓

hash signature

-------------------------------------

2. Multi-table LSH

Use several hash tables to reduce collision probability.

Parameters:

num_tables
hash_size

-------------------------------------

3. Index storage

Buckets should store:

track_id
fingerprint_id

-------------------------------------

4. Candidate retrieval

Given query fingerprint:

Return candidate track IDs.

-------------------------------------

Integration

Add function:

search_candidates(fingerprint)

Used before similarity scoring.

-------------------------------------

Testing

Add tests:

tests/test_lsh_index.py

Test cases:

• identical fingerprints
• transposed melodies
• random fingerprints
• large dataset simulation

-------------------------------------

Performance goals

Index 10k MIDI files in <30 seconds

Query time < 50 ms

-------------------------------------

Engineering rules

Python 3.11+

Must pass existing CI tests

Add docstrings and comments

Follow architecture documentation

-------------------------------------

Expected result

Working LSH indexing engine integrated into Mantra prototype.