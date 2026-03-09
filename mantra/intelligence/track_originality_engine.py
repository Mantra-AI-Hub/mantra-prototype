"""Track originality scoring for the MANTRA anti-plagiarism pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class TrackFingerprint:
    track_id: str
    melody_vector: list[float]
    harmony_vector: list[float]
    rhythm_vector: list[float]
    energy_vector: list[float]


@dataclass(frozen=True)
class OriginalityResult:
    track_id: str
    similarity_score: float
    originality_score: float
    most_similar_track: str | None


class TrackOriginalityEngine:
    """Estimate track originality from deterministic multi-signal fingerprints."""

    def extract_fingerprint(self, track_data: Mapping[str, object]) -> TrackFingerprint:
        track_id = str(track_data.get("track_id", "unknown_track"))
        feature_vector = self._coerce_vector(track_data.get("feature_vector"))
        melody_vector = self._resolve_signal(track_data, "melody_vector", fallback=self._build_melody(track_data, feature_vector))
        harmony_vector = self._resolve_signal(track_data, "harmony_vector", fallback=self._build_harmony(track_data, feature_vector))
        rhythm_vector = self._resolve_signal(track_data, "rhythm_vector", fallback=self._build_rhythm(track_data, feature_vector))
        energy_vector = self._resolve_signal(track_data, "energy_vector", fallback=self._build_energy(track_data, feature_vector))
        return TrackFingerprint(
            track_id=track_id,
            melody_vector=melody_vector,
            harmony_vector=harmony_vector,
            rhythm_vector=rhythm_vector,
            energy_vector=energy_vector,
        )

    def compare_fingerprints(self, fp1: TrackFingerprint, fp2: TrackFingerprint) -> dict[str, float]:
        return {
            "melody_similarity": self._cosine_similarity(fp1.melody_vector, fp2.melody_vector),
            "harmony_similarity": self._cosine_similarity(fp1.harmony_vector, fp2.harmony_vector),
            "rhythm_similarity": self._cosine_similarity(fp1.rhythm_vector, fp2.rhythm_vector),
            "energy_similarity": self._cosine_similarity(fp1.energy_vector, fp2.energy_vector),
        }

    def compute_similarity(self, fp1: TrackFingerprint, fp2: TrackFingerprint) -> float:
        components = self.compare_fingerprints(fp1, fp2)
        similarity = (
            components["melody_similarity"] * 0.4
            + components["harmony_similarity"] * 0.3
            + components["rhythm_similarity"] * 0.2
            + components["energy_similarity"] * 0.1
        )
        similarity = float(np.clip(similarity, 0.0, 1.0))
        if similarity >= 1.0 - 1e-12:
            return 1.0
        if similarity <= 1e-12:
            return 0.0
        return similarity

    def compute_originality(
        self,
        track_fp: TrackFingerprint,
        library: Sequence[TrackFingerprint],
    ) -> OriginalityResult:
        match = self.find_most_similar(track_fp, library)
        if match is None:
            return OriginalityResult(
                track_id=track_fp.track_id,
                similarity_score=0.0,
                originality_score=1.0,
                most_similar_track=None,
            )
        most_similar_fp, similarity = match
        originality = float(np.clip(1.0 - similarity, 0.0, 1.0))
        return OriginalityResult(
            track_id=track_fp.track_id,
            similarity_score=float(np.clip(similarity, 0.0, 1.0)),
            originality_score=originality,
            most_similar_track=most_similar_fp.track_id,
        )

    def find_most_similar(
        self,
        track_fp: TrackFingerprint,
        library: Sequence[TrackFingerprint],
    ) -> tuple[TrackFingerprint, float] | None:
        best_match: TrackFingerprint | None = None
        best_score = -1.0
        for candidate in library:
            if candidate.track_id == track_fp.track_id:
                continue
            score = self.compute_similarity(track_fp, candidate)
            if score > best_score:
                best_match = candidate
                best_score = score
        if best_match is None:
            return None
        return best_match, float(np.clip(best_score, 0.0, 1.0))

    def is_risky(self, result: OriginalityResult, threshold: float = 0.8) -> bool:
        return result.similarity_score >= threshold

    def _resolve_signal(self, track_data: Mapping[str, object], key: str, fallback: list[float]) -> list[float]:
        explicit = self._coerce_vector(track_data.get(key))
        return explicit if explicit else fallback

    def _build_melody(self, track_data: Mapping[str, object], feature_vector: list[float]) -> list[float]:
        if feature_vector:
            return self._resample(feature_vector, 8)
        return [
            self._ratio(track_data.get("tempo"), 180.0),
            self._ratio(track_data.get("mood"), 1.0),
            self._ratio(track_data.get("novelty"), 1.0),
            self._ratio(track_data.get("energy"), 1.0),
        ]

    def _build_harmony(self, track_data: Mapping[str, object], feature_vector: list[float]) -> list[float]:
        if feature_vector:
            return self._resample(feature_vector[::-1], 8)
        emotional = self._ratio(
            track_data.get("emotional_intensity"),
            1.0,
            default=(self._ratio(track_data.get("energy"), 1.0) + self._ratio(track_data.get("mood"), 1.0)) / 2.0,
        )
        return [
            self._ratio(track_data.get("harmonic_density"), 1.0),
            self._ratio(track_data.get("mood"), 1.0),
            emotional,
            self._ratio(track_data.get("novelty"), 1.0),
        ]

    def _build_rhythm(self, track_data: Mapping[str, object], feature_vector: list[float]) -> list[float]:
        if feature_vector:
            rolled = np.roll(np.asarray(feature_vector, dtype=float), 1)
            return self._resample(rolled.tolist(), 8)
        return [
            self._ratio(track_data.get("tempo"), 180.0),
            self._ratio(track_data.get("rhythm_complexity"), 1.0),
            self._ratio(track_data.get("energy"), 1.0),
            self._ratio(track_data.get("novelty"), 1.0),
        ]

    def _build_energy(self, track_data: Mapping[str, object], feature_vector: list[float]) -> list[float]:
        if feature_vector:
            vector = np.abs(np.asarray(feature_vector, dtype=float))
            return self._resample(vector.tolist(), 8)
        emotional = self._ratio(
            track_data.get("emotional_intensity"),
            1.0,
            default=(self._ratio(track_data.get("energy"), 1.0) + self._ratio(track_data.get("mood"), 1.0)) / 2.0,
        )
        return [
            self._ratio(track_data.get("energy"), 1.0),
            emotional,
            self._ratio(track_data.get("mood"), 1.0),
            self._ratio(track_data.get("harmonic_density"), 1.0),
        ]

    def _cosine_similarity(self, left: Sequence[float], right: Sequence[float]) -> float:
        left_vec = np.asarray(left, dtype=float)
        right_vec = np.asarray(right, dtype=float)
        if left_vec.size == 0 or right_vec.size == 0:
            return 0.0
        if left_vec.shape == right_vec.shape and np.allclose(left_vec, right_vec):
            return 1.0
        target_size = int(max(left_vec.size, right_vec.size))
        left_aligned = np.asarray(self._resample(left_vec.tolist(), target_size), dtype=float)
        right_aligned = np.asarray(self._resample(right_vec.tolist(), target_size), dtype=float)
        left_norm = float(np.linalg.norm(left_aligned))
        right_norm = float(np.linalg.norm(right_aligned))
        if left_norm == 0.0 or right_norm == 0.0:
            return 0.0
        cosine = float(np.dot(left_aligned, right_aligned) / (left_norm * right_norm))
        bounded = (cosine + 1.0) / 2.0
        return float(np.clip(bounded, 0.0, 1.0))

    def _resample(self, values: Sequence[float], target_size: int) -> list[float]:
        if target_size <= 0:
            return []
        vector = np.asarray(list(values), dtype=float)
        if vector.size == 0:
            return [0.0] * target_size
        if vector.size == target_size:
            return vector.astype(float).tolist()
        if vector.size == 1:
            return [float(vector[0])] * target_size
        source = np.linspace(0.0, 1.0, num=vector.size, dtype=float)
        target = np.linspace(0.0, 1.0, num=target_size, dtype=float)
        return np.interp(target, source, vector).astype(float).tolist()

    def _coerce_vector(self, value: object) -> list[float]:
        if value is None:
            return []
        try:
            array = np.asarray(value, dtype=float).reshape(-1)
        except (TypeError, ValueError):
            return []
        if array.size == 0:
            return []
        return array.astype(float).tolist()

    def _ratio(self, value: object, scale: float, default: float = 0.0) -> float:
        if value is None:
            return float(np.clip(default, 0.0, 1.0))
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return float(np.clip(default, 0.0, 1.0))
        if scale == 0:
            return float(np.clip(numeric, 0.0, 1.0))
        return float(np.clip(numeric / scale, 0.0, 1.0))
