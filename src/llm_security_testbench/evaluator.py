"""Evaluation orchestration."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from llm_security_testbench import __version__
from llm_security_testbench.metrics import all_slices, binary_metrics, paired_analysis
from llm_security_testbench.models import EvaluationRecord, Example, Prediction
from llm_security_testbench.predictors import Predictor


class EvaluationError(RuntimeError):
    """Raised when an evaluation cannot produce a trustworthy report."""


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    """Evaluation report plus row-level records and prediction errors."""

    report: dict[str, Any]
    records: tuple[EvaluationRecord, ...]
    errors: tuple[dict[str, str], ...]


def _label_counts(examples: Sequence[Example]) -> dict[str, int]:
    return {
        "benign": sum(example.label == 0 for example in examples),
        "attack": sum(example.label == 1 for example in examples),
    }


def evaluate(
    examples: Sequence[Example],
    predictor: Predictor,
    *,
    dataset_source: str,
    split: str,
    threshold: float = 0.5,
    max_workers: int = 1,
    allow_missing: bool = False,
) -> EvaluationResult:
    """Evaluate a predictor against labeled examples."""

    if not examples:
        raise EvaluationError("cannot evaluate an empty dataset")
    if not 0.0 <= threshold <= 1.0:
        raise EvaluationError("threshold must be between 0 and 1")

    predictions = predictor.predict_many(examples, max_workers=max_workers)
    records: list[EvaluationRecord] = []
    errors: list[dict[str, str]] = []

    for example in examples:
        prediction = predictions.get(
            example.id,
            Prediction(id=example.id, error="predictor returned no result"),
        )
        if prediction.error:
            errors.append({"id": example.id, "error": prediction.error})
            continue
        try:
            predicted_label = prediction.resolved_label(threshold)
        except ValueError as exc:
            errors.append({"id": example.id, "error": str(exc)})
            continue
        records.append(
            EvaluationRecord(
                example=example,
                prediction=prediction,
                predicted_label=predicted_label,
            )
        )

    if errors and not allow_missing:
        preview = "; ".join(f"{item['id']}: {item['error']}" for item in errors[:3])
        suffix = "" if len(errors) <= 3 else f"; and {len(errors) - 3} more"
        raise EvaluationError(
            f"{len(errors)} predictions failed ({preview}{suffix}); "
            "use --allow-missing only when partial coverage is intentional"
        )
    if not records:
        raise EvaluationError("no valid predictions were produced")

    report: dict[str, Any] = {
        "schema_version": "1.0",
        "tool": {"name": "llm-security-testbench", "version": __version__},
        "run": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "dataset_source": dataset_source,
            "split": split,
            "predictor": predictor.name,
            "threshold": threshold,
        },
        "dataset": {
            "examples": len(examples),
            "label_counts": _label_counts(examples),
            "paired_rows": sum(example.pair_id is not None for example in examples),
        },
        "coverage": {
            "evaluated": len(records),
            "failed": len(errors),
            "rate": len(records) / len(examples),
        },
        "metrics": binary_metrics(records),
        "paired_analysis": paired_analysis(records),
        "slices": all_slices(records),
        "errors": errors,
    }
    return EvaluationResult(report=report, records=tuple(records), errors=tuple(errors))
