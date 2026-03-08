"""Streaming dataset scanner for very large ingestion jobs."""

from __future__ import annotations

from pathlib import Path
from typing import Generator


def stream_audio_files(path: str) -> Generator[str, None, None]:
    root = Path(path)
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"Dataset folder not found: {path}")

    for file_path in root.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() == ".wav":
            yield str(file_path)
