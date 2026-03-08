"""Task implementations for background audio indexing."""

from __future__ import annotations

import io
import wave
from datetime import datetime, timezone
from typing import Dict, List, Tuple

import numpy as np

from mantra.audio_engine.feature_extractor import (
    extract_chroma,
    extract_pitch_contour,
    extract_spectral_features,
    extract_tempo,
)
from mantra.fingerprint_engine.audio_fingerprint import generate_fingerprint
from mantra.vector_engine.embedding_builder import build_music_embedding


AudioData = Tuple[np.ndarray, int]
Fingerprint = List[Tuple[str, int]]


def _decode_wav_bytes(raw_bytes: bytes) -> AudioData:
    if not raw_bytes:
        raise ValueError("Empty audio payload")

    with wave.open(io.BytesIO(raw_bytes), "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_rate = wav_file.getframerate()
        sample_width = wav_file.getsampwidth()
        frames = wav_file.readframes(wav_file.getnframes())

    if sample_width == 1:
        dtype = np.uint8
        offset = 128.0
        scale = 128.0
    elif sample_width == 2:
        dtype = np.int16
        offset = 0.0
        scale = 32768.0
    elif sample_width == 4:
        dtype = np.int32
        offset = 0.0
        scale = 2147483648.0
    else:
        raise ValueError(f"Unsupported WAV sample width: {sample_width} bytes")

    audio = np.frombuffer(frames, dtype=dtype).astype(np.float32)
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)

    audio = (audio - offset) / scale
    return audio.astype(np.float32, copy=False), int(sample_rate)


def embedding_task(audio: AudioData) -> np.ndarray:
    """Build embedding from audio payload."""
    features = {
        "chroma": extract_chroma(audio),
        "tempo": extract_tempo(audio),
        "pitch_contour": extract_pitch_contour(audio),
        "spectral": extract_spectral_features(audio),
    }
    return build_music_embedding(features)


def fingerprint_task(audio: AudioData) -> Fingerprint:
    """Build fingerprint from audio payload."""
    return generate_fingerprint(audio)


def index_audio_task(payload: Dict[str, object]) -> Dict[str, object]:
    """Compute artifacts required for audio indexing."""
    audio_bytes = bytes(payload["audio_bytes"])
    track_id = str(payload["track_id"])
    filename = str(payload["filename"])
    embedding_path = str(payload["embedding_path"])
    artist = str(payload.get("artist") or "")
    album = str(payload.get("album") or "")
    genre = str(payload.get("genre") or "")
    tags = payload.get("tags") or []
    year = payload.get("year")

    audio = _decode_wav_bytes(audio_bytes)
    embedding = embedding_task(audio)
    fingerprint = fingerprint_task(audio)
    duration = float(len(audio[0]) / max(1, int(audio[1])))

    return {
        "track_id": track_id,
        "filename": filename,
        "duration": duration,
        "embedding_path": embedding_path,
        "fingerprint_hash_count": int(len(fingerprint)),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "artist": artist,
        "album": album,
        "genre": genre,
        "tags": tags,
        "year": int(year) if year is not None else None,
        "embedding": embedding,
        "fingerprint": fingerprint,
    }
