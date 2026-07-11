# Promptfoo Integration

Promptfoo evaluates complete LLM applications and supports many providers, assertions,
and red-team strategies. LLM Security Testbench focuses on binary prompt-injection
detectors and pair-aware classification metrics. The export connects these two scopes
without pretending they are identical.

## Export

```bash
uv run llmst export-promptfoo \
  --dataset hf://3nesdeniz/turkish-conversation-prompt-injection \
  --split test \
  --output-dir integrations/promptfoo
```

Generated files:

| File | Purpose |
| --- | --- |
| `promptfooconfig.yaml` | Promptfoo configuration using `{{text}}` |
| `tests.json` | One test per dataset row with a deterministic expected label |
| `provider.py` | HTTP provider that normalizes the detector response |
| `RUN.md` | Dataset-specific run instructions |

## Run

```bash
cd integrations/promptfoo
export TESTBENCH_ENDPOINT=https://guardrail.example/v1/classify
export TESTBENCH_TOKEN='replace-me' # optional
npx promptfoo@latest eval -c promptfooconfig.yaml
npx promptfoo@latest view
```

The provider expects a response label at `label`. Override a nested field with:

```bash
export TESTBENCH_LABEL_FIELD=result.label
```

## Reading the results

Promptfoo shows each case and deterministic pass/fail status. Use that interface for
application-level review and comparison across providers.

Use `llmst evaluate` for confusion metrics, pair accuracy, attack-family false positives,
and the privacy-minimized report format. Promptfoo's exported `results.json` is not yet
treated as the canonical metric input because its schema and application-level semantics
can vary by Promptfoo version and assertion configuration.

## Security note

The exported tests contain the raw dataset text. Treat the export according to the
dataset's license and sensitivity. The testbench's own `report.json`, `report.md`, and
`predictions.jsonl` omit raw text by default.
