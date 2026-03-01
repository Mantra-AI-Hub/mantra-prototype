from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class Metadata:
    """
    Basic metadata for a Creative Fingerprint.
    """

    fingerprint_version: str
    created_at: datetime
    source_name: str | None = None
    composer: str | None = None
    genre: str | None = None
