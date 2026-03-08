from pathlib import Path

from mantra.explain_engine.similarity_explainer import explain_similarity


def test_explain_similarity_returns_structured_result():
    a = Path("test.mid")
    b = Path("test_transposed.mid")

    result = explain_similarity(str(a), str(b))

    assert set(result.keys()) >= {
        "similarity_score",
        "shared_interval_patterns",
        "rhythm_similarity",
        "pitch_similarity",
    }
    assert 0.0 <= result["similarity_score"] <= 1.0
    assert result["pitch_similarity"] > 0.9
    assert result["rhythm_similarity"] > 0.9
    assert result["shared_interval_patterns"]
