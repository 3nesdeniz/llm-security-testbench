"""Dataset loading, Hugging Face caching, and structural validation."""

from __future__ import annotations

import hashlib
import json
import re
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llm_security_testbench.models import Example, ModelError


class DatasetError(RuntimeError):
    """Raised when a dataset cannot be loaded or validated."""


@dataclass(frozen=True, slots=True)
class ValidationResult:
    """Structural validation findings for a dataset."""

    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    total_examples: int
    label_counts: dict[int, int]
    split_counts: dict[str, int]
    pair_count: int

    @property
    def valid(self) -> bool:
        return not self.errors


def _parse_jsonl(content: str, *, origin: str) -> list[Example]:
    examples: list[Example] = []
    for line_number, line in enumerate(content.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DatasetError(f"{origin}:{line_number}: invalid JSON: {exc.msg}") from exc
        if not isinstance(payload, dict):
            raise DatasetError(f"{origin}:{line_number}: each row must be a JSON object")
        try:
            examples.append(Example.from_mapping(payload))
        except ModelError as exc:
            raise DatasetError(f"{origin}:{line_number}: {exc}") from exc
    if not examples:
        raise DatasetError(f"{origin}: no examples found")
    return examples


def _load_jsonl_file(path: Path) -> list[Example]:
    try:
        content = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise DatasetError(f"could not read dataset file {path}: {exc}") from exc
    return _parse_jsonl(content, origin=str(path))


def _hf_parts(source: str, revision: str) -> tuple[str, str]:
    identifier = source.removeprefix("hf://")
    if "@" in identifier:
        identifier, revision = identifier.rsplit("@", 1)
    if identifier.count("/") != 1 or not all(identifier.split("/")):
        raise DatasetError("Hugging Face sources must use hf://owner/dataset[@revision]")
    return identifier, revision


def _hf_cache_path(cache_dir: Path, repo_id: str, revision: str, split: str) -> Path:
    digest = hashlib.sha256(f"{repo_id}@{revision}:{split}".encode()).hexdigest()[:16]
    safe_name = re.sub(r"[^a-zA-Z0-9_.-]+", "-", repo_id)
    return cache_dir / f"{safe_name}-{split}-{digest}.jsonl"


def _download_hf_split(
    repo_id: str,
    *,
    revision: str,
    split: str,
    cache_dir: Path,
    refresh: bool,
) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    destination = _hf_cache_path(cache_dir, repo_id, revision, split)
    if destination.exists() and not refresh:
        return destination

    encoded_repo = urllib.parse.quote(repo_id, safe="/")
    encoded_revision = urllib.parse.quote(revision, safe="")
    url = (
        f"https://huggingface.co/datasets/{encoded_repo}/resolve/"
        f"{encoded_revision}/data/{split}.jsonl?download=true"
    )
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "llm-security-testbench/0.1.0"},
    )
    try:
        # The request URL is constructed from a fixed HTTPS Hugging Face origin.
        with urllib.request.urlopen(request, timeout=30) as response:  # nosec B310
            content = response.read()
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
        if destination.exists():
            return destination
        raise DatasetError(f"could not download {repo_id} split {split!r}: {exc}") from exc

    temporary = destination.with_suffix(".tmp")
    temporary.write_bytes(content)
    temporary.replace(destination)
    return destination


def _directory_files(path: Path, split: str) -> list[Path]:
    split_names = ("train", "validation", "test") if split == "all" else (split,)
    files: list[Path] = []
    for split_name in split_names:
        candidates = (path / "data" / f"{split_name}.jsonl", path / f"{split_name}.jsonl")
        selected = next((candidate for candidate in candidates if candidate.is_file()), None)
        if selected is None:
            raise DatasetError(
                f"could not find {split_name}.jsonl under {path} or {path / 'data'}"
            )
        files.append(selected)
    return files


def load_dataset(
    source: str,
    *,
    split: str = "test",
    revision: str = "main",
    cache_dir: Path | None = None,
    refresh: bool = False,
) -> list[Example]:
    """Load examples from local JSONL, a dataset directory, or Hugging Face."""

    if split not in {"train", "validation", "test", "all"}:
        raise DatasetError("split must be train, validation, test, or all")

    examples: list[Example] = []
    if source.startswith("hf://"):
        repo_id, revision = _hf_parts(source, revision)
        resolved_cache = cache_dir or Path.home() / ".cache" / "llm-security-testbench"
        split_names = ("train", "validation", "test") if split == "all" else (split,)
        for split_name in split_names:
            path = _download_hf_split(
                repo_id,
                revision=revision,
                split=split_name,
                cache_dir=resolved_cache,
                refresh=refresh,
            )
            examples.extend(_load_jsonl_file(path))
    else:
        path = Path(source).expanduser().resolve()
        if path.is_dir():
            for file_path in _directory_files(path, split):
                examples.extend(_load_jsonl_file(file_path))
        elif path.is_file():
            examples = _load_jsonl_file(path)
            if split != "all":
                examples = [example for example in examples if example.split == split]
        else:
            raise DatasetError(f"dataset source does not exist: {path}")

    if not examples:
        raise DatasetError(f"dataset source contains no examples for split {split!r}")
    return examples


def _count_by(values: Iterable[Any]) -> dict[Any, int]:
    counts: dict[Any, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return counts


def validate_examples(examples: list[Example]) -> ValidationResult:
    """Validate IDs, normalized text uniqueness, labels, and pair integrity."""

    errors: list[str] = []
    warnings: list[str] = []

    id_counts = _count_by(example.id for example in examples)
    duplicate_ids = sorted(value for value, count in id_counts.items() if count > 1)
    if duplicate_ids:
        errors.append(f"duplicate example ids: {', '.join(duplicate_ids[:10])}")

    normalized_text_counts = _count_by(
        " ".join(example.text.casefold().split()) for example in examples
    )
    duplicate_text_count = sum(1 for count in normalized_text_counts.values() if count > 1)
    if duplicate_text_count:
        errors.append(f"duplicate normalized texts: {duplicate_text_count}")

    pair_groups: dict[str, list[Example]] = {}
    for example in examples:
        if example.pair_id:
            pair_groups.setdefault(example.pair_id, []).append(example)

    for pair_id, rows in sorted(pair_groups.items()):
        if len(rows) != 2:
            errors.append(f"pair {pair_id} has {len(rows)} rows; expected 2")
            continue
        if {row.label for row in rows} != {0, 1}:
            errors.append(f"pair {pair_id} must contain one benign and one attack row")
        if len({row.split for row in rows}) != 1:
            errors.append(f"pair {pair_id} crosses dataset splits")

    if not pair_groups:
        warnings.append("dataset contains no paired benign/attack examples")
    if not any(example.label == 1 for example in examples):
        warnings.append("dataset contains no positive examples")
    if not any(example.label == 0 for example in examples):
        warnings.append("dataset contains no negative examples")

    return ValidationResult(
        errors=tuple(errors),
        warnings=tuple(warnings),
        total_examples=len(examples),
        label_counts=dict(sorted(_count_by(example.label for example in examples).items())),
        split_counts=dict(sorted(_count_by(example.split for example in examples).items())),
        pair_count=len(pair_groups),
    )
