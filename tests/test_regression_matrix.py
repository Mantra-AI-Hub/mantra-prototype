import json
import pytest
from pathlib import Path

from core.fingerprint import build_fingerprint_from_pitch
from core.similarity import calculate_similarity


DATA_PATH = Path("data/regression_cases.json")


def load_cases():
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.mark.parametrize("case", load_cases())
def test_regression_matrix(case):
    f1 = build_fingerprint_from_pitch(case["a"])
    f2 = build_fingerprint_from_pitch(case["b"])

    result = calculate_similarity(f1, f2)

    assert round(result, 4) == round(case["expected"], 4)