from __future__ import annotations

from pathlib import Path

import pytest

from llm_security_testbench.datasets import load_dataset
from llm_security_testbench.evaluator import evaluate
from llm_security_testbench.predictors import JsonlPredictor, PythonPredictor

FIXTURES = Path(__file__).parent / "fixtures"


def test_evaluation_computes_confusion_and_pair_metrics() -> None:
    examples = load_dataset(str(FIXTURES / "dataset.jsonl"), split="test")
    predictor = JsonlPredictor(FIXTURES / "predictions.jsonl")
    result = evaluate(
        examples,
        predictor,
        dataset_source="fixture",
        split="test",
    )

    metrics = result.report["metrics"]
    assert metrics["tp"] == 1
    assert metrics["fp"] == 1
    assert metrics["tn"] == 5
    assert metrics["fn"] == 1
    assert metrics["accuracy"] == pytest.approx(0.75)
    assert metrics["balanced_accuracy"] == pytest.approx(2 / 3)
    assert metrics["false_positive_rate"] == pytest.approx(1 / 6)
    assert metrics["roc_auc"] == pytest.approx(11 / 12)

    pairs = result.report["paired_analysis"]
    assert pairs["complete_pairs"] == 2
    assert pairs["both_correct"] == 1
    assert pairs["both_wrong"] == 1
    assert pairs["pair_accuracy"] == pytest.approx(0.5)

    family_slices = result.report["slices"]["by_attack_family"]
    assert set(family_slices) == {
        "direct_instruction_override",
        "system_prompt_extraction",
    }
    assert family_slices["direct_instruction_override"]["count"] == 2


def test_python_predictor_loads_file_function() -> None:
    examples = load_dataset(str(FIXTURES / "dataset.jsonl"), split="test")
    predictor = PythonPredictor(f"{FIXTURES / 'predictor.py'}:predict")
    result = evaluate(
        examples,
        predictor,
        dataset_source="fixture",
        split="test",
        max_workers=2,
    )
    assert result.report["metrics"]["accuracy"] == 1.0
