from __future__ import annotations

import json
import urllib.error
from pathlib import Path

import pytest

from llm_security_testbench import datasets
from llm_security_testbench.datasets import DatasetError, load_dataset, validate_examples
from llm_security_testbench.models import Example


def _row(row_id: str, split: str, *, label: int = 0, pair_id: str | None = None) -> str:
    return json.dumps(
        {
            "id": row_id,
            "text": f"text for {row_id}",
            "label": label,
            "pair_id": pair_id,
            "split": split,
        }
    )


def test_load_directory_with_all_splits(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    for split in ("train", "validation", "test"):
        (data_dir / f"{split}.jsonl").write_text(
            _row(f"row-{split}", split) + "\n",
            encoding="utf-8",
        )

    examples = load_dataset(str(tmp_path), split="all")
    assert {example.split for example in examples} == {"train", "validation", "test"}


class _Response:
    def __init__(self, content: bytes) -> None:
        self.content = content

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *args: object) -> None:
        del args

    def read(self) -> bytes:
        return self.content


def test_hugging_face_download_revision_and_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    content = (_row("remote-test", "test") + "\n").encode()
    requested_urls: list[str] = []

    def fake_urlopen(request: object, timeout: int) -> _Response:
        assert timeout == 30
        requested_urls.append(request.full_url)
        return _Response(content)

    monkeypatch.setattr(datasets.urllib.request, "urlopen", fake_urlopen)
    examples = load_dataset(
        "hf://owner/repository@v1.2.3",
        split="test",
        cache_dir=tmp_path,
        refresh=True,
    )
    assert examples[0].id == "remote-test"
    assert "/v1.2.3/data/test.jsonl" in requested_urls[0]

    def failing_urlopen(request: object, timeout: int) -> _Response:
        del request, timeout
        raise urllib.error.URLError("offline")

    monkeypatch.setattr(datasets.urllib.request, "urlopen", failing_urlopen)
    cached = load_dataset(
        "hf://owner/repository@v1.2.3",
        split="test",
        cache_dir=tmp_path,
        refresh=True,
    )
    assert cached[0].id == "remote-test"


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("not-json\n", "invalid JSON"),
        ("[]\n", "must be a JSON object"),
        ('{"id":"x","text":"y"}\n', "missing label"),
        ("\n", "no examples found"),
    ],
)
def test_invalid_jsonl_rows_raise(tmp_path: Path, content: str, message: str) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(DatasetError, match=message):
        load_dataset(str(path), split="all")


def test_invalid_dataset_sources_raise(tmp_path: Path) -> None:
    with pytest.raises(DatasetError, match="split must"):
        load_dataset(str(tmp_path), split="invalid")
    with pytest.raises(DatasetError, match="does not exist"):
        load_dataset(str(tmp_path / "missing"), split="test")
    with pytest.raises(DatasetError, match="could not find test.jsonl"):
        load_dataset(str(tmp_path), split="test")
    with pytest.raises(DatasetError, match="hf://owner/dataset"):
        load_dataset("hf://invalid", split="test", cache_dir=tmp_path)


def test_validation_reports_pair_and_class_warnings() -> None:
    benign_only = [Example(id="one", text="one", label=0, split="test")]
    warning_result = validate_examples(benign_only)
    assert "dataset contains no paired benign/attack examples" in warning_result.warnings
    assert "dataset contains no positive examples" in warning_result.warnings

    broken_pairs = [
        Example(id="a", text="a", label=0, pair_id="pair", split="test"),
        Example(id="b", text="b", label=0, pair_id="pair", split="train"),
    ]
    broken_result = validate_examples(broken_pairs)
    assert any("one benign and one attack" in error for error in broken_result.errors)
    assert any("crosses dataset splits" in error for error in broken_result.errors)

    incomplete = validate_examples(
        [Example(id="c", text="c", label=1, pair_id="incomplete", split="test")]
    )
    assert any("expected 2" in error for error in incomplete.errors)
