from __future__ import annotations

import json
from pathlib import Path

from llm_security_testbench.datasets import load_dataset
from llm_security_testbench.promptfoo import export_promptfoo

FIXTURES = Path(__file__).parent / "fixtures"


def test_promptfoo_export_contains_tests_and_provider(tmp_path: Path) -> None:
    examples = load_dataset(str(FIXTURES / "dataset.jsonl"), split="test")
    paths = export_promptfoo(
        examples,
        tmp_path,
        dataset_source="fixture",
        split="test",
    )

    tests = json.loads(paths["tests"].read_text(encoding="utf-8"))
    assert len(tests) == 8
    assert tests[0]["assert"][0]["type"] == "equals"
    assert {test["assert"][0]["value"] for test in tests} == {"attack", "benign"}
    assert "file://provider.py" in paths["config"].read_text(encoding="utf-8")
    assert "TESTBENCH_ENDPOINT" in paths["provider"].read_text(encoding="utf-8")
