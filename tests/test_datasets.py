from __future__ import annotations

from pathlib import Path

from llm_security_testbench.datasets import load_dataset, validate_examples

FIXTURES = Path(__file__).parent / "fixtures"


def test_load_and_validate_jsonl_fixture() -> None:
    examples = load_dataset(str(FIXTURES / "dataset.jsonl"), split="test")
    result = validate_examples(examples)

    assert result.valid
    assert result.total_examples == 8
    assert result.label_counts == {0: 6, 1: 2}
    assert result.split_counts == {"test": 8}
    assert result.pair_count == 2


def test_validation_detects_duplicate_id() -> None:
    examples = load_dataset(str(FIXTURES / "dataset.jsonl"), split="test")
    result = validate_examples([*examples, examples[0]])
    assert not result.valid
    assert any("duplicate example ids" in error for error in result.errors)
