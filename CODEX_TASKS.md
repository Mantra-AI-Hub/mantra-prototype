# Mantra AI Development Tasks

Project: Mantra Prototype

Repository:
https://github.com/Mantra-AI-Hub/mantra-prototype

Goal:
Build an AI-powered music similarity engine capable of detecting melodic plagiarism across large MIDI datasets.

Current implemented components:

• Interval fingerprint engine
• Similarity scoring engine
• SQLite persistence layer
• CLI interface (mantra.py)
• Regression test suite
• GitHub Actions CI pipeline

All tests are passing.

Next development goals:

----------------------------------------

1. LSH Index Engine

Implement Locality Sensitive Hashing for fast similarity search.

Requirements:

• Index interval fingerprints
• Support large datasets (100k+ MIDI files)
• Provide fast candidate retrieval

Files to create:

indexing/lsh_index.py
indexing/hash_functions.py
indexing/bucket_store.py

----------------------------------------

2. Dataset Indexer

Create dataset indexing pipeline.

Features:

• Index folder of MIDI files
• Generate fingerprints
• Store hashes in SQLite
• Update existing index

Files:

dataset/index_dataset.py

----------------------------------------

3. Explainable Similarity

Add explanation layer.

Engine should return:

• matched intervals
• similarity score
• match confidence
• explanation JSON

Files:

analysis/explain_similarity.py

----------------------------------------

4. REST API

Add REST API for remote analysis.

Endpoints:

POST /analyze

Input:
MIDI file

Output:
similarity matches

Files:

api/server.py
api/routes.py

----------------------------------------

5. Performance Benchmark

Add benchmark script.

Test:

• indexing speed
• search speed
• memory usage

Files:

benchmarks/benchmark_search.py

----------------------------------------

Engineering rules:

• Python 3.11+
• Must pass all existing tests
• Add tests for new modules
• Follow project architecture