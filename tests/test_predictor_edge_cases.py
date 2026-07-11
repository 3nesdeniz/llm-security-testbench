from __future__ import annotations

from pathlib import Path

import pytest

from llm_security_testbench.evaluator import EvaluationError, evaluate
from llm_security_testbench.models import Example, ModelError, Prediction
from llm_security_testbench.predictors import (
    HttpPredictor,
    JsonlPredictor,
    PredictorError,
    PythonPredictor,
)


def test_jsonl_predictor_reports_missing_prediction(tmp_path: Path) -> None:
    path = tmp_path / "predictions.jsonl"
    path.write_text('{"id":"one","prediction":"benign"}\n', encoding="utf-8")
    predictor = JsonlPredictor(path)
    examples = [
        Example(id="one", text="one", label=0),
        Example(id="two", text="two", label=1),
    ]

    with pytest.raises(EvaluationError, match="predictions failed"):
        evaluate(examples, predictor, dataset_source="fixture", split="test")

    partial = evaluate(
        examples,
        predictor,
        dataset_source="fixture",
        split="test",
        allow_missing=True,
    )
    assert partial.report["coverage"]["evaluated"] == 1
    assert partial.report["coverage"]["failed"] == 1


@pytest.mark.parametrize(
    "content",
    [
        "not-json\n",
        "[]\n",
        '{"prediction":"attack"}\n',
        '{"id":"one"}\n',
        '{"id":"one","prediction":"attack"}\n{"id":"one","prediction":"benign"}\n',
        '{"id":"one","prediction":"maybe"}\n',
    ],
)
def test_invalid_prediction_files_raise(tmp_path: Path, content: str) -> None:
    path = tmp_path / "invalid.jsonl"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(PredictorError):
        JsonlPredictor(path)


def test_python_predictor_contract_errors(tmp_path: Path) -> None:
    with pytest.raises(PredictorError, match="must use"):
        PythonPredictor("missing-spec")
    with pytest.raises(PredictorError, match="does not exist"):
        PythonPredictor(f"{tmp_path / 'missing.py'}:predict")

    path = tmp_path / "predictor.py"
    path.write_text(
        "def wrong_name(example):\n    return 0\n\n"
        "def explode(example):\n    raise RuntimeError('failure')\n",
        encoding="utf-8",
    )
    with pytest.raises(PredictorError, match="function not found"):
        PythonPredictor(f"{path}:predict")

    predictor = PythonPredictor(f"{path}:explode")
    result = predictor.predict_many([Example(id="one", text="one", label=0)])
    assert result["one"].error == "failure"


def test_http_predictor_validates_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(PredictorError, match="http"):
        HttpPredictor("file:///tmp/model")
    with pytest.raises(PredictorError, match="positive"):
        HttpPredictor("https://example.com", timeout=0)
    with pytest.raises(PredictorError, match="negative"):
        HttpPredictor("https://example.com", retries=-1)
    monkeypatch.delenv("MISSING_TOKEN", raising=False)
    with pytest.raises(PredictorError, match="not set"):
        HttpPredictor("https://example.com", token_env="MISSING_TOKEN")


def test_prediction_and_evaluation_validation() -> None:
    with pytest.raises(ModelError, match="between 0 and 1"):
        Prediction(id="one", score=1.2)
    with pytest.raises(ModelError, match="label"):
        Prediction(id="one", label=2)
    with pytest.raises(EvaluationError, match="empty"):
        evaluate([], JsonlPredictor.__new__(JsonlPredictor), dataset_source="x", split="test")

    class _Predictor:
        name = "fixed"

        def predict_many(self, examples: object, *, max_workers: int = 1) -> dict[str, Prediction]:
            del examples, max_workers
            return {"one": Prediction(id="one", label=0)}

    with pytest.raises(EvaluationError, match="threshold"):
        evaluate(
            [Example(id="one", text="one", label=0)],
            _Predictor(),
            dataset_source="x",
            split="test",
            threshold=2,
        )
