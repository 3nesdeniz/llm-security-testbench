from __future__ import annotations

import json
from pathlib import Path

from llm_security_testbench.datasets import load_dataset
from llm_security_testbench.evaluator import evaluate
from llm_security_testbench.predictors import JsonlPredictor
from llm_security_testbench.reporting import write_reports

FIXTURES = Path(__file__).parent / "fixtures"


def test_reports_omit_raw_prompt_text(tmp_path: Path) -> None:
    examples = load_dataset(str(FIXTURES / "dataset.jsonl"), split="test")
    result = evaluate(
        examples,
        JsonlPredictor(FIXTURES / "predictions.jsonl"),
        dataset_source="fixture",
        split="test",
    )
    paths = write_reports(result, tmp_path)

    markdown = paths["markdown"].read_text(encoding="utf-8")
    rows = paths["predictions"].read_text(encoding="utf-8")
    report = json.loads(paths["json"].read_text(encoding="utf-8"))

    assert "Önceki talimatları" not in markdown
    assert "Önceki talimatları" not in rows
    assert report["metrics"]["fp"] == 1
    assert "Paired boundary analysis" in markdown
