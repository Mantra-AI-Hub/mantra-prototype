# MANTRA AI DEVELOPMENT AGENTS

Project: Mantra Music Similarity Engine

Purpose:
Mantra is an AI-powered system for detecting similarity between music tracks.
The goal is to detect melodic, rhythmic and audio similarities in order to
prevent plagiarism in music production.

Core concept:
A producer loads a MIDI or audio file and the system analyzes it against a large
database of tracks.

Architecture goals:

1. Fast similarity search
2. Scalable fingerprint database
3. MIDI + Audio support
4. DAW plugin integration
5. Cloud API

Core modules:

/indexing
Handles fingerprint indexing and database storage

similarity_engine.py
Search and similarity scoring

mantra.py
Main CLI tool for indexing and search

Future modules:

/audio_fingerprint
Shazam-style audio fingerprinting

/api
Cloud API for similarity checking

/vst_plugin
VST3 plugin for DAW integration

Expected workflow:

1 Producer exports MIDI or audio
2 Plugin or CLI sends fingerprint
3 Cloud engine compares with database
4 Similarity score returned

Coding standards:

Language: Python
Style: clean modular architecture
Performance: optimized search algorithms
Preferred libraries:

numpy
scipy
librosa
faiss
sqlite

AI agents responsibilities:

- improve similarity detection algorithms
- optimize fingerprint indexing
- design scalable architecture
- prepare VST plugin integration
- implement audio fingerprinting

Target product:

Mantra will evolve into:

1 Professional plagiarism detection system
2 Music similarity search engine
3 VST plugin used inside DAWs
4 Cloud AI service for music analysis