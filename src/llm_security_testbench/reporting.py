"""Write machine-readable and reviewer-friendly evaluation reports."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from llm_security_testbench.evaluator import EvaluationResult


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _percentage(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value) * 100:.2f}%"


def _number(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.2f}"


def _metric_table(metrics: Mapping[str, Any]) -> list[str]:
    return [
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Evaluated examples | {metrics['count']} |",
        f"| True positives | {metrics['tp']} |",
        f"| False positives | {metrics['fp']} |",
        f"| True negatives | {metrics['tn']} |",
        f"| False negatives | {metrics['fn']} |",
        f"| Accuracy | {_percentage(metrics['accuracy'])} |",
        f"| Balanced accuracy | {_percentage(metrics['balanced_accuracy'])} |",
        f"| Precision | {_percentage(metrics['precision'])} |",
        f"| Attack recall | {_percentage(metrics['recall'])} |",
        f"| Specificity | {_percentage(metrics['specificity'])} |",
        f"| F1 | {_percentage(metrics['f1'])} |",
        f"| False-positive rate | {_percentage(metrics['false_positive_rate'])} |",
        f"| False-negative rate | {_percentage(metrics['false_negative_rate'])} |",
        f"| ROC AUC | {_number(metrics['roc_auc'])} |",
        f"| Mean latency | {_number(metrics['mean_latency_ms'])} ms |",
    ]


def _slice_table(title: str, slices: Mapping[str, Mapping[str, Any]]) -> list[str]:
    lines = [
        f"### {title}",
        "",
        "| Slice | N | Recall | FPR | Balanced accuracy | Pair-relevant precision |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, metrics in slices.items():
        lines.append(
            f"| `{name}` | {metrics['count']} | {_percentage(metrics['recall'])} | "
            f"{_percentage(metrics['false_positive_rate'])} | "
            f"{_percentage(metrics['balanced_accuracy'])} | "
            f"{_percentage(metrics['precision'])} |"
        )
    if not slices:
        lines.append("| _No slices_ | 0 | n/a | n/a | n/a | n/a |")
    return lines


def render_markdown(report: Mapping[str, Any]) -> str:
    """Render a deterministic Markdown report without raw prompt text."""

    run = report["run"]
    coverage = report["coverage"]
    pairs = report["paired_analysis"]
    lines = [
        "# LLM Security Testbench Report",
        "",
        f"- **Created:** {run['created_at']}",
        f"- **Dataset:** `{run['dataset_source']}`",
        f"- **Split:** `{run['split']}`",
        f"- **Predictor:** `{run['predictor']}`",
        f"- **Decision threshold:** `{run['threshold']}`",
        f"- **Coverage:** {coverage['evaluated']}/{report['dataset']['examples']} "
        f"({_percentage(coverage['rate'])})",
        "",
        "## Overall classification",
        "",
        *_metric_table(report["metrics"]),
        "",
        "## Paired boundary analysis",
        "",
        "A pair passes only when both the attack and its matched legitimate request are "
        "classified correctly.",
        "",
        "| Outcome | Pairs |",
        "| --- | ---: |",
        f"| Complete pairs | {pairs['complete_pairs']} |",
        f"| Both sides correct | {pairs['both_correct']} |",
        f"| Attack correct, benign wrong | {pairs['attack_only_correct']} |",
        f"| Benign correct, attack wrong | {pairs['benign_only_correct']} |",
        f"| Both sides wrong | {pairs['both_wrong']} |",
        f"| Pair accuracy | {_percentage(pairs['pair_accuracy'])} |",
        "",
        "## Sliced results",
        "",
        *_slice_table("Attack family", report["slices"]["by_attack_family"]),
        "",
        *_slice_table("Category", report["slices"]["by_category"]),
        "",
        *_slice_table("Source context", report["slices"]["by_source_context"]),
        "",
        "## Interpretation boundary",
        "",
        "This report measures behavior on the selected dataset and split. It is not a "
        "production certification and does not establish security against unseen attacks.",
    ]
    if report["errors"]:
        lines.extend(
            [
                "",
                "## Prediction errors",
                "",
                f"{len(report['errors'])} rows were excluded. "
                "See `report.json` for IDs and errors.",
            ]
        )
    return "\n".join(lines) + "\n"


def write_reports(result: EvaluationResult, output_dir: Path) -> dict[str, Path]:
    """Write summary JSON, Markdown, and row-level prediction JSONL."""

    resolved = output_dir.expanduser().resolve()
    report_json = resolved / "report.json"
    report_markdown = resolved / "report.md"
    predictions_jsonl = resolved / "predictions.jsonl"

    _atomic_write(
        report_json,
        json.dumps(result.report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    _atomic_write(report_markdown, render_markdown(result.report))

    prediction_lines: list[str] = []
    for record in result.records:
        row = {
            "id": record.example.id,
            "truth": record.example.label,
            "prediction": record.predicted_label,
            "score": record.prediction.score,
            "correct": record.correct,
            "category": record.example.category,
            "attack_family": record.example.attack_family,
            "source_context": record.example.source_context,
            "pair_id": record.example.pair_id,
            "split": record.example.split,
            "latency_ms": record.prediction.latency_ms,
        }
        prediction_lines.append(json.dumps(row, ensure_ascii=False, sort_keys=True))
    _atomic_write(predictions_jsonl, "\n".join(prediction_lines) + "\n")

    return {
        "json": report_json,
        "markdown": report_markdown,
        "predictions": predictions_jsonl,
    }
