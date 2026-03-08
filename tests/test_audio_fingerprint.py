import numpy as np

from mantra.fingerprint_engine.audio_fingerprint import generate_fingerprint, match_fingerprint


def _make_tone(freq: float, duration: float = 2.0, sample_rate: int = 22050, lead_silence: float = 0.0):
    t = np.linspace(0.0, duration, int(sample_rate * duration), endpoint=False, dtype=np.float32)
    tone = 0.5 * np.sin(2.0 * np.pi * freq * t)
    if lead_silence > 0:
        silence = np.zeros(int(sample_rate * lead_silence), dtype=np.float32)
        tone = np.concatenate([silence, tone])
    return tone.astype(np.float32), sample_rate


def test_generate_fingerprint_returns_landmarks():
    audio = _make_tone(440.0)
    fingerprint = generate_fingerprint(audio)

    assert fingerprint
    assert isinstance(fingerprint[0][0], str)
    assert isinstance(fingerprint[0][1], int)


def test_match_fingerprint_prefers_same_track_with_time_shift():
    base_audio = _make_tone(440.0)
    shifted_audio = _make_tone(440.0, lead_silence=0.5)
    different_audio = _make_tone(660.0)

    query_fp = generate_fingerprint(shifted_audio)
    db = {
        "track_a": generate_fingerprint(base_audio),
        "track_b": generate_fingerprint(different_audio),
    }

    results = match_fingerprint(query_fp, db)

    assert results
    assert results[0][0] == "track_a"
    if len(results) > 1:
        assert results[0][1] >= results[1][1]


def test_match_fingerprint_empty_query_returns_empty():
    assert match_fingerprint([], {"x": []}) == []
