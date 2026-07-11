"""Command-line interface."""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

from llm_security_testbench import __version__
from llm_security_testbench.datasets import DatasetError, load_dataset, validate_examples
from llm_security_testbench.evaluator import EvaluationError, evaluate
from llm_security_testbench.predictors import (
    HttpPredictor,
    JsonlPredictor,
    Predictor,
    PredictorError,
    PythonPredictor,
)
from llm_security_testbench.promptfoo import export_promptfoo
from llm_security_testbench.reporting import write_reports


def _dataset_arguments(parser: argparse.ArgumentParser, *, default_split: str) -> None:
    parser.add_argument(
        "--dataset",
        required=True,
        help="Local JSONL/directory or hf://owner/dataset[@revision]",
    )
    parser.add_argument(
        "--split",
        choices=("train", "validation", "test", "all"),
        default=default_split,
    )
    parser.add_argument("--revision", default="main", help="Hugging Face revision")
    parser.add_argument("--refresh", action="store_true", help="Refresh the HF cache")


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""

    parser = argparse.ArgumentParser(
        prog="llmst",
        description="Evaluate prompt-injection detectors and LLM guardrails reproducibly.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser("validate", help="Validate dataset structure")
    _dataset_arguments(validate_parser, default_split="all")

    evaluate_parser = subparsers.add_parser("evaluate", help="Run a detector evaluation")
    _dataset_arguments(evaluate_parser, default_split="test")
    evaluate_parser.add_argument("--threshold", type=float, default=0.5)
    evaluate_parser.add_argument("--max-workers", type=int, default=4)
    evaluate_parser.add_argument("--allow-missing", action="store_true")
    evaluate_parser.add_argument(
        "--output-dir",
        type=Path,
        help="Report directory; defaults to reports/<UTC timestamp>",
    )

    predictor_group = evaluate_parser.add_mutually_exclusive_group(required=True)
    predictor_group.add_argument("--predictions", type=Path, help="Offline prediction JSONL")
    predictor_group.add_argument("--python", dest="python_spec", help="path.py:function")
    predictor_group.add_argument("--http", dest="http_endpoint", help="Detector endpoint")

    evaluate_parser.add_argument("--http-label-field", default="label")
    evaluate_parser.add_argument("--http-score-field", default="score")
    evaluate_parser.add_argument("--http-text-field", default="text")
    evaluate_parser.add_argument("--http-timeout", type=float, default=15.0)
    evaluate_parser.add_argument("--http-retries", type=int, default=1)
    evaluate_parser.add_argument("--http-token-env")
    evaluate_parser.add_argument(
        "--http-header",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Additional request header; repeatable",
    )

    promptfoo_parser = subparsers.add_parser(
        "export-promptfoo",
        help="Export Promptfoo tests and an HTTP provider",
    )
    _dataset_arguments(promptfoo_parser, default_split="test")
    promptfoo_parser.add_argument("--output-dir", type=Path, required=True)

    return parser


def _headers(values: Sequence[str]) -> dict[str, str]:
    headers: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise PredictorError(f"invalid HTTP header {value!r}; expected KEY=VALUE")
        key, header_value = value.split("=", 1)
        if not key.strip():
            raise PredictorError("HTTP header name cannot be empty")
        headers[key.strip()] = header_value
    return headers


def _predictor(args: argparse.Namespace) -> Predictor:
    if args.predictions:
        return JsonlPredictor(args.predictions)
    if args.python_spec:
        return PythonPredictor(args.python_spec)
    if args.http_endpoint:
        return HttpPredictor(
            args.http_endpoint,
            label_field=args.http_label_field,
            score_field=args.http_score_field,
            text_field=args.http_text_field,
            timeout=args.http_timeout,
            retries=args.http_retries,
            token_env=args.http_token_env,
            headers=_headers(args.http_header),
        )
    raise PredictorError("one predictor option is required")


def _default_output_dir() -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path("reports") / timestamp


def _run_validate(args: argparse.Namespace) -> int:
    examples = load_dataset(
        args.dataset,
        split=args.split,
        revision=args.revision,
        refresh=args.refresh,
    )
    result = validate_examples(examples)
    print(f"Examples: {result.total_examples}")
    print(f"Labels: {result.label_counts}")
    print(f"Splits: {result.split_counts}")
    print(f"Pairs: {result.pair_count}")
    for warning in result.warnings:
        print(f"Warning: {warning}")
    for error in result.errors:
        print(f"Error: {error}", file=sys.stderr)
    print("Dataset is valid." if result.valid else "Dataset validation failed.")
    return 0 if result.valid else 1


def _run_evaluate(args: argparse.Namespace) -> int:
    examples = load_dataset(
        args.dataset,
        split=args.split,
        revision=args.revision,
        refresh=args.refresh,
    )
    validation = validate_examples(examples)
    if not validation.valid:
        raise DatasetError("dataset failed validation: " + "; ".join(validation.errors))

    result = evaluate(
        examples,
        _predictor(args),
        dataset_source=args.dataset,
        split=args.split,
        threshold=args.threshold,
        max_workers=args.max_workers,
        allow_missing=args.allow_missing,
    )
    paths = write_reports(result, args.output_dir or _default_output_dir())
    metrics = result.report["metrics"]
    print(f"Evaluated: {metrics['count']} examples")
    print(f"TP/FP/TN/FN: {metrics['tp']}/{metrics['fp']}/{metrics['tn']}/{metrics['fn']}")
    if metrics["balanced_accuracy"] is not None:
        print(f"Balanced accuracy: {metrics['balanced_accuracy']:.4f}")
    if metrics["false_positive_rate"] is not None:
        print(f"False-positive rate: {metrics['false_positive_rate']:.4f}")
    print(f"Markdown report: {paths['markdown']}")
    print(f"JSON report: {paths['json']}")
    return 0


def _run_promptfoo(args: argparse.Namespace) -> int:
    examples = load_dataset(
        args.dataset,
        split=args.split,
        revision=args.revision,
        refresh=args.refresh,
    )
    validation = validate_examples(examples)
    if not validation.valid:
        raise DatasetError("dataset failed validation: " + "; ".join(validation.errors))
    paths = export_promptfoo(
        examples,
        args.output_dir,
        dataset_source=args.dataset,
        split=args.split,
    )
    print(f"Exported {len(examples)} Promptfoo tests.")
    print(f"Config: {paths['config']}")
    print(f"Instructions: {paths['instructions']}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""

    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "validate":
            return _run_validate(args)
        if args.command == "evaluate":
            return _run_evaluate(args)
        if args.command == "export-promptfoo":
            return _run_promptfoo(args)
        parser.error(f"unknown command: {args.command}")
    except (DatasetError, EvaluationError, PredictorError) as exc:
        if os.environ.get("LLMST_DEBUG"):
            raise
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    return 2
