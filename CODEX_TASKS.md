# Mantra AI Development Tasks

Project: Mantra Prototype

Goal:
Build a scalable music similarity engine capable of detecting melodic plagiarism across large MIDI datasets.

Current status:
- Interval fingerprint engine implemented
- Similarity scoring implemented
- SQLite persistence
- Regression tests
- CI pipeline working

Next engineering goals:

1. Build scalable fingerprint index
Implement Locality Sensitive Hashing (LSH) for fast search across large MIDI datasets.

2. Implement dataset indexing pipeline
Allow indexing of thousands of MIDI files.

3. Add explainable similarity output
Return structural explanation of similarity matches.

4. Build REST API endpoint
POST /analyze
Upload MIDI → return similarity matches.

5. Prepare architecture for DAW plugin integration.

Constraints:
- Python 3.11+
- Must pass all existing tests
- Add new tests for new functionality