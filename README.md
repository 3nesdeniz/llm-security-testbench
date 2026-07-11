# LLM Security Testbench

**Reproducible, pair-aware evaluation for prompt-injection detectors and LLM guardrails.**

LLM Security Testbench measures two requirements at the same time: whether a defense
detects attacks and whether it leaves legitimate user requests alone. It produces a
confusion matrix, false-positive and false-negative rates, per-family slices, and a
strict matched-pair score for benign and malicious requests that share similar language.

The first supported corpus is the open
[Turkish Conversation Prompt-Injection Dataset](https://huggingface.co/datasets/3nesdeniz/turkish-conversation-prompt-injection).
The engine itself is dataset- and detector-agnostic.

## Why this exists

Attack-only evaluation rewards systems that block aggressively. A detector can report
high attack recall while disrupting ordinary users, security teams, and agent workflows.
That is not a production-ready control.

This testbench keeps the operational trade-off visible:

- TP, FP, TN, and FN are reported together;
- attack recall never appears without false-positive rate and specificity;
- matched benign/attack pairs receive a separate all-or-nothing score;
- results are sliced by attack family, source context, category, and split;
- reports contain row IDs and metrics, but omit raw prompt text by default.

## Capabilities

- Load JSONL datasets from disk or directly from Hugging Face.
- Validate IDs, labels, duplicate text, split isolation, and pair integrity.
- Evaluate precomputed predictions, Python callables, or JSON HTTP endpoints.
- Accept binary labels, attack probabilities, or both.
- Run HTTP and Python predictors concurrently.
- Export machine-readable JSON, reviewer-friendly Markdown, and row-level JSONL.
- Export complete Promptfoo tests plus a ready HTTP provider.
- Cache remote datasets and pin an explicit Hugging Face revision.

## Installation

Python 3.10 or newer is required. The runtime package has no third-party dependencies.

```bash
git clone https://github.com/3nesdeniz/llm-security-testbench.git
cd llm-security-testbench
uv sync --dev
uv run llmst --version
```

## Validate a dataset

Validate every published split directly from Hugging Face:

```bash
uv run llmst validate \
  --dataset hf://3nesdeniz/turkish-conversation-prompt-injection \
  --split all
```

Local dataset directories may contain either `data/train.jsonl`,
`data/validation.jsonl`, and `data/test.jsonl`, or the same files at the directory root.

## Evaluate offline predictions

Prediction files use one JSON object per line:

```json
{"id":"tcpi_p0129_a","prediction":"attack","score":0.93}
{"id":"tcpi_p0129_b","prediction":"benign","score":0.08}
```

`prediction` accepts `attack`/`benign`, `unsafe`/`safe`, booleans, or `1`/`0`.
`score` is optional and represents attack probability between `0` and `1`.

```bash
uv run llmst evaluate \
  --dataset hf://3nesdeniz/turkish-conversation-prompt-injection \
  --split test \
  --predictions predictions.jsonl \
  --output-dir reports/offline-run
```

Missing predictions fail the run by default. Use `--allow-missing` only when partial
coverage is intentional; the report will show the excluded IDs and coverage rate.

## Evaluate a Python detector

Expose a function that accepts the full dataset row as a dictionary:

```python
def predict(example):
    result = detector.classify(example["text"])
    return {
        "label": result.label,
        "score": result.attack_probability,
    }
```

Then run:

```bash
uv run llmst evaluate \
  --dataset hf://3nesdeniz/turkish-conversation-prompt-injection \
  --split test \
  --python path/to/detector.py:predict \
  --max-workers 4 \
  --output-dir reports/python-run
```

## Evaluate an HTTP guardrail

The default request contract is:

```json
{
  "text": "input to classify",
  "id": "stable-row-id",
  "metadata": {
    "category": "benign_boundary",
    "attack_family": "none",
    "source_context": "direct_user",
    "pair_id": "pair_0084",
    "split": "test"
  }
}
```

The default response contract accepts a label, a score, or both:

```json
{"label":"benign","score":0.14}
```

```bash
export GUARDRAIL_TOKEN='replace-me'

uv run llmst evaluate \
  --dataset hf://3nesdeniz/turkish-conversation-prompt-injection \
  --split test \
  --http https://guardrail.example/v1/classify \
  --http-token-env GUARDRAIL_TOKEN \
  --max-workers 8 \
  --output-dir reports/http-run
```

Nested responses are supported with `--http-label-field result.label` and
`--http-score-field result.score`. See [the endpoint contract](docs/endpoint-contract.md)
for retries, headers, and label normalization.

## Reports

Every completed run writes:

| File | Purpose |
| --- | --- |
| `report.json` | Full machine-readable metrics, slices, run metadata, and errors |
| `report.md` | Human review report with confusion, pair, and family tables |
| `predictions.jsonl` | Row IDs, truth, prediction, score, slice metadata, and latency |

Raw prompt text is intentionally excluded. This makes reports safer to move through CI,
ticketing, and review systems. The source dataset remains the canonical place to inspect
individual examples.

## Promptfoo integration

Export the selected split as Promptfoo tests with deterministic `equals` assertions:

```bash
uv run llmst export-promptfoo \
  --dataset hf://3nesdeniz/turkish-conversation-prompt-injection \
  --split test \
  --output-dir integrations/promptfoo

cd integrations/promptfoo
export TESTBENCH_ENDPOINT=https://guardrail.example/v1/classify
npx promptfoo@latest eval -c promptfooconfig.yaml
npx promptfoo@latest view
```

The export includes `promptfooconfig.yaml`, `tests.json`, a Python HTTP provider, and
run instructions. See [Promptfoo integration](docs/promptfoo.md) for the exact boundary
between Promptfoo's application-level evaluation and this package's detector metrics.

## Metrics

The report includes:

- confusion counts: TP, FP, TN, FN;
- accuracy and balanced accuracy;
- precision, attack recall, specificity, and F1;
- false-positive and false-negative rates;
- ROC AUC when every evaluated row includes an attack score;
- mean predictor latency;
- pair accuracy, where both the attack and matched legitimate request must be correct;
- per-family, category, source-context, and split slices.

Metric definitions and zero-denominator behavior are documented in
[docs/metrics.md](docs/metrics.md).

## Dataset contract

Only `id`, `text`, and `label` are required. The following optional fields enable richer
slicing and paired analysis:

| Field | Purpose |
| --- | --- |
| `category` | Dataset composition slice |
| `attack_family` | Positive-row attack taxonomy |
| `source_context` | Direct user, retrieved document, tool output, memory, and similar contexts |
| `pair_id` | Links one attack with one legitimate boundary case |
| `source_type` | Provenance metadata |
| `split` | Train, validation, test, or a custom split name |

Unknown fields are preserved and passed to Python predictors.

## Security and interpretation boundaries

- This is an evaluation tool, not a production guardrail.
- A successful run is not proof of security against unseen or adaptive attacks.
- Synthetic test distributions do not estimate real-world attack prevalence.
- Scores are interpreted as attack probabilities; reverse-scored APIs need an adapter.
- Dataset licensing and privacy remain the responsibility of the dataset publisher and user.
- Only test systems you own or are explicitly authorized to assess.

See [SECURITY.md](SECURITY.md) for vulnerability reporting.

## Türkçe özet

LLM Security Testbench, prompt injection savunmalarını yalnızca yakalanan saldırılar
üzerinden değerlendirmez. Meşru kullanıcı taleplerindeki yanlış alarmları, eşleştirilmiş
güvenli/saldırı çiftlerini ve saldırı ailesi bazındaki sonuçları aynı raporda gösterir.
İlk desteklenen veri seti Türkçe Conversation Prompt-Injection Dataset'tir; araç farklı
JSONL veri setleri ve detector API'leriyle de çalışır.

## Citation

If this software supports published work, cite [CITATION.cff](CITATION.cff) and the
dataset used for the run. Dataset results should always include the exact revision,
split, threshold, and predictor configuration.

## License

Code is licensed under [Apache License 2.0](LICENSE). Dataset licenses are independent;
the Turkish Conversation Prompt-Injection Dataset is published under CC BY 4.0.

Maintained by [Enes Deniz](https://github.com/3nesdeniz), Co-Founder of
[AltaySec](https://altaysec.com.tr/).
