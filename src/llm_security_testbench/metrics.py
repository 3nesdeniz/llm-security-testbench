"""Binary classification, slicing, and paired-boundary metrics."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Sequence
from typing import Any

from llm_security_testbench.models import EvaluationRecord


def _safe_div(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


def _roc_auc(records: Sequence[EvaluationRecord]) -> float | None:
    if not records:
        return None
    scored_records: list[tuple[float, int]] = []
    for record in records:
        if record.prediction.score is None:
            return None
        scored_records.append((record.prediction.score, record.example.label))
    positives = sum(record.example.label == 1 for record in records)
    negatives = len(records) - positives
    if positives == 0 or negatives == 0:
        return None

    ordered = sorted(scored_records, key=lambda item: item[0])
    rank_sum = 0.0
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][0] == ordered[index][0]:
            end += 1
        average_rank = ((index + 1) + end) / 2
        rank_sum += average_rank * sum(label == 1 for _, label in ordered[index:end])
        index = end
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def binary_metrics(records: Sequence[EvaluationRecord]) -> dict[str, int | float | None]:
    """Compute confusion counts and standard binary classification metrics."""

    true_positive = sum(
        record.example.label == 1 and record.predicted_label == 1 for record in records
    )
    false_positive = sum(
        record.example.label == 0 and record.predicted_label == 1 for record in records
    )
    true_negative = sum(
        record.example.label == 0 and record.predicted_label == 0 for record in records
    )
    false_negative = sum(
        record.example.label == 1 and record.predicted_label == 0 for record in records
    )

    recall = _safe_div(true_positive, true_positive + false_negative)
    specificity = _safe_div(true_negative, true_negative + false_positive)
    precision = _safe_div(true_positive, true_positive + false_positive)
    accuracy = _safe_div(true_positive + true_negative, len(records))
    f1 = (
        _safe_div(2 * precision * recall, precision + recall)
        if precision is not None and recall is not None
        else None
    )
    balanced_accuracy = (
        (recall + specificity) / 2
        if recall is not None and specificity is not None
        else None
    )
    latencies = [
        record.prediction.latency_ms
        for record in records
        if record.prediction.latency_ms is not None
    ]

    return {
        "count": len(records),
        "tp": true_positive,
        "fp": false_positive,
        "tn": true_negative,
        "fn": false_negative,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "false_positive_rate": _safe_div(false_positive, false_positive + true_negative),
        "false_negative_rate": _safe_div(false_negative, false_negative + true_positive),
        "roc_auc": _roc_auc(records),
        "mean_latency_ms": _safe_div(sum(latencies), len(latencies)),
    }


def metric_slices(
    records: Sequence[EvaluationRecord],
    key_function: Callable[[EvaluationRecord], str | None],
) -> dict[str, dict[str, int | float | None]]:
    """Compute binary metrics for each non-empty slice key."""

    grouped: dict[str, list[EvaluationRecord]] = defaultdict(list)
    for record in records:
        key = key_function(record)
        if key:
            grouped[key].append(record)
    return {key: binary_metrics(grouped[key]) for key in sorted(grouped)}


def paired_analysis(records: Sequence[EvaluationRecord]) -> dict[str, int | float | None]:
    """Measure whether both sides of each benign/attack pair are classified correctly."""

    grouped: dict[str, list[EvaluationRecord]] = defaultdict(list)
    for record in records:
        if record.example.pair_id:
            grouped[record.example.pair_id].append(record)

    complete_pairs = [
        pair
        for pair in grouped.values()
        if len(pair) == 2 and {record.example.label for record in pair} == {0, 1}
    ]
    both_correct = sum(all(record.correct for record in pair) for pair in complete_pairs)
    both_wrong = sum(all(not record.correct for record in pair) for pair in complete_pairs)
    attack_only_correct = 0
    benign_only_correct = 0
    for pair in complete_pairs:
        attack = next(record for record in pair if record.example.label == 1)
        benign = next(record for record in pair if record.example.label == 0)
        if attack.correct and not benign.correct:
            attack_only_correct += 1
        if benign.correct and not attack.correct:
            benign_only_correct += 1

    return {
        "complete_pairs": len(complete_pairs),
        "both_correct": both_correct,
        "attack_only_correct": attack_only_correct,
        "benign_only_correct": benign_only_correct,
        "both_wrong": both_wrong,
        "pair_accuracy": _safe_div(both_correct, len(complete_pairs)),
    }


def attack_family_map(records: Sequence[EvaluationRecord]) -> dict[str, str]:
    """Map pair IDs to the attack family of their positive row."""

    result: dict[str, str] = {}
    for record in records:
        if record.example.label == 1 and record.example.pair_id:
            result[record.example.pair_id] = record.example.attack_family
    return result


def all_slices(records: Sequence[EvaluationRecord]) -> dict[str, Any]:
    """Build category, source-context, split, and pair-aware family slices."""

    family_by_pair = attack_family_map(records)

    def resolved_family(record: EvaluationRecord) -> str | None:
        if record.example.label == 1:
            return record.example.attack_family
        if record.example.pair_id:
            return family_by_pair.get(record.example.pair_id)
        return None

    return {
        "by_category": metric_slices(records, lambda record: record.example.category),
        "by_source_context": metric_slices(
            records, lambda record: record.example.source_context
        ),
        "by_split": metric_slices(records, lambda record: record.example.split),
        "by_attack_family": metric_slices(records, resolved_family),
    }
