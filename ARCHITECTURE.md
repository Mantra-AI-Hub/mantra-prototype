# Mantra Architecture

Mantra is a structural music similarity engine designed to analyze melodic patterns inside MIDI data.

Unlike audio fingerprinting systems, Mantra focuses on **melodic interval structures**.

---

# System Overview

Mantra pipeline:

MIDI File  
↓  
Pitch Extraction  
↓  
Interval Normalization  
↓  
Fingerprint Generation  
↓  
Similarity Search  
↓  
Similarity Score

---

# Core Modules

## extraction

Responsible for reading MIDI files and extracting note pitch sequences.

Key tasks:

- parse MIDI events
- extract note pitches
- preserve melodic order

---

## core_algorithms

Contains the fundamental music analysis algorithms.

Main algorithm:

Interval normalization.

Example:

Melody:

60 62 64 65

Intervals:

+2 +2 +1

This makes the melody **key invariant**.

---

## fingerprint

Transforms interval sequences into compact fingerprint vectors.

Purpose:

- efficient comparison
- hashing
- indexing

---

## indexing

Implements fast search structures.

Used for:

- storing fingerprints
- retrieving candidate matches

Possible future algorithms:

- LSH (Locality Sensitive Hashing)
- ANN search
- vector similarity

---

## matching

Responsible for computing similarity scores.

Methods include:

- interval overlap
- structural similarity
- regression testing

---

## persistence

Handles storage of fingerprints.

Current implementation:

SQLite database.

Future:

- distributed database
- vector database
- cloud storage

---

## app

FastAPI service exposing Mantra functionality through HTTP API.

Endpoints allow:

- MIDI upload
- similarity search
- fingerprint inspection

---

# Data Flow

Example workflow:

1. User uploads MIDI file
2. System extracts note pitches
3. Intervals are computed
4. Fingerprint is generated
5. Fingerprint compared against database
6. Similar tracks returned

---

# Why Interval Fingerprinting

Interval representation provides:

Key invariance  
Tempo independence  
Structural similarity detection  

Example:

C major melody

C D E G

Transposed melody

F G A C

Both produce identical interval structures.

---

# Future Architecture

Next stages of Mantra development:

Phase 1  
MIDI structural similarity

Phase 2  
Audio fingerprinting

Phase 3  
Large-scale similarity search

Phase 4  
Cloud API

Phase 5  
DAW plugin (VST3)

---

# Long-Term Vision

Mantra aims to become a professional tool for:

music producers  
record labels  
music publishers  
copyright analysis  

The system may evolve into a hybrid platform combining:

- structural music analysis
- AI similarity detection
- audio fingerprinting