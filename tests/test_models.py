from __future__ import annotations

import pytest

from llm_security_testbench.models import Example, ModelError, Prediction, normalize_label


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (1, 1),
        (0, 0),
        (True, 1),
        (False, 0),
        ("attack", 1),
        ("prompt injection", 1),
        ("safe", 0),
        ("allowed", 0),
    ],
)
def test_normalize_label(value: object, expected: int) -> None:
    assert normalize_label(value) == expected


def test_normalize_label_rejects_unknown_value() -> None:
    with pytest.raises(ModelError):
        normalize_label("maybe")


def test_example_keeps_unknown_metadata() -> None:
    example = Example.from_mapping(
        {"id": "row-1", "text": "test", "label": 0, "custom": "value"}
    )
    assert example.extra == {"custom": "value"}
    assert example.to_mapping()["custom"] == "value"


def test_score_prediction_resolves_at_threshold() -> None:
    prediction = Prediction.from_value("row-1", 0.71)
    assert prediction.resolved_label(0.7) == 1
    assert prediction.resolved_label(0.8) == 0
