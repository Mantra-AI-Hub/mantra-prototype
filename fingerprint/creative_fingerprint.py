from dataclasses import dataclass

from mantra.fingerprint.metadata import Metadata
from mantra.fingerprint.melody_features import MelodyFeatures


@dataclass(frozen=True)
class CreativeFingerprint:
    """
    Top-level fingerprint container (V1).

    Combines metadata and melodic analysis.
    """

    metadata: Metadata
    melody: MelodyFeatures
