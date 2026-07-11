"""Export labeled detector tests for Promptfoo."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from llm_security_testbench.models import Example

_PROVIDER_SOURCE = '''"""Promptfoo provider for a binary prompt-injection detector endpoint."""

import json
import os
import urllib.request


def _normalize_label(value):
    normalized = str(value).strip().casefold().replace("-", "_").replace(" ", "_")
    if normalized in {"1", "attack", "blocked", "malicious", "prompt_injection", "unsafe", "true"}:
        return "attack"
    if normalized in {"0", "allowed", "benign", "false", "safe"}:
        return "benign"
    raise ValueError(f"unsupported detector label: {value!r}")


def _nested(payload, path):
    current = payload
    for component in path.split("."):
        current = current[component]
    return current


def call_api(prompt, options, context):
    endpoint = os.environ["TESTBENCH_ENDPOINT"]
    text_field = os.getenv("TESTBENCH_TEXT_FIELD", "text")
    label_field = os.getenv("TESTBENCH_LABEL_FIELD", "label")
    payload = {
        text_field: prompt,
        "id": context.get("vars", {}).get("id"),
        "metadata": context.get("test", {}).get("metadata", {}),
    }
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    token = os.getenv("TESTBENCH_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        result = json.loads(response.read().decode("utf-8"))
    return {"output": _normalize_label(_nested(result, label_field))}
'''


def export_promptfoo(
    examples: Sequence[Example],
    output_dir: Path,
    *,
    dataset_source: str,
    split: str,
) -> dict[str, Path]:
    """Export Promptfoo config, tests, endpoint provider, and run instructions."""

    resolved = output_dir.expanduser().resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    tests_path = resolved / "tests.json"
    config_path = resolved / "promptfooconfig.yaml"
    provider_path = resolved / "provider.py"
    instructions_path = resolved / "RUN.md"

    tests = []
    for example in examples:
        expected = "attack" if example.label == 1 else "benign"
        tests.append(
            {
                "description": example.id,
                "vars": {"id": example.id, "text": example.text},
                "metadata": {
                    "category": example.category,
                    "attack_family": example.attack_family,
                    "source_context": example.source_context,
                    "pair_id": example.pair_id,
                    "split": example.split,
                },
                "assert": [
                    {
                        "type": "equals",
                        "value": expected,
                        "metric": "prompt-injection-classification",
                    }
                ],
            }
        )

    tests_path.write_text(
        json.dumps(tests, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    config_path.write_text(
        "\n".join(
            [
                "# yaml-language-server: $schema=https://promptfoo.dev/config-schema.json",
                f'description: "LLM Security Testbench export: {split}"',
                "prompts:",
                '  - "{{text}}"',
                "providers:",
                '  - id: "file://provider.py"',
                '    label: "Binary prompt-injection detector"',
                'tests: "file://tests.json"',
                'outputPath: "results.json"',
                "",
            ]
        ),
        encoding="utf-8",
    )
    provider_path.write_text(_PROVIDER_SOURCE, encoding="utf-8")
    instructions_path.write_text(
        "\n".join(
            [
                "# Run the Promptfoo export",
                "",
                f"Dataset: `{dataset_source}`  ",
                f"Split: `{split}`  ",
                f"Examples: `{len(examples)}`",
                "",
                "Set the detector endpoint and run Promptfoo:",
                "",
                "```bash",
                "export TESTBENCH_ENDPOINT=https://your-detector.example/v1/classify",
                "# Optional: export TESTBENCH_TOKEN=...",
                "npx promptfoo@latest eval -c promptfooconfig.yaml",
                "npx promptfoo@latest view",
                "```",
                "",
                "The endpoint must return a JSON label at `label` by default. Override the nested "
                "response path with `TESTBENCH_LABEL_FIELD`, for example `result.label`.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return {
        "config": config_path,
        "tests": tests_path,
        "provider": provider_path,
        "instructions": instructions_path,
    }
