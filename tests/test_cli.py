from __future__ import annotations

from pathlib import Path

from llm_security_testbench.cli import main

FIXTURES = Path(__file__).parent / "fixtures"


def test_cli_validate_and_evaluate(tmp_path: Path) -> None:
    dataset = str(FIXTURES / "dataset.jsonl")
    assert main(["validate", "--dataset", dataset, "--split", "test"]) == 0
    assert (
        main(
            [
                "evaluate",
                "--dataset",
                dataset,
                "--split",
                "test",
                "--predictions",
                str(FIXTURES / "predictions.jsonl"),
                "--output-dir",
                str(tmp_path),
            ]
        )
        == 0
    )
    assert (tmp_path / "report.json").is_file()
    assert (tmp_path / "report.md").is_file()
    assert (tmp_path / "predictions.jsonl").is_file()


def test_cli_python_export_and_error_paths(tmp_path: Path) -> None:
    dataset = str(FIXTURES / "dataset.jsonl")
    predictor = f"{FIXTURES / 'predictor.py'}:predict"
    report_dir = tmp_path / "python-report"
    assert (
        main(
            [
                "evaluate",
                "--dataset",
                dataset,
                "--split",
                "test",
                "--python",
                predictor,
                "--output-dir",
                str(report_dir),
            ]
        )
        == 0
    )

    promptfoo_dir = tmp_path / "promptfoo"
    assert (
        main(
            [
                "export-promptfoo",
                "--dataset",
                dataset,
                "--split",
                "test",
                "--output-dir",
                str(promptfoo_dir),
            ]
        )
        == 0
    )
    assert (promptfoo_dir / "promptfooconfig.yaml").is_file()

    assert main(["validate", "--dataset", str(tmp_path / "missing")]) == 2
