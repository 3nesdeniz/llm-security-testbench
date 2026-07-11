# HTTP Endpoint Contract

The HTTP predictor evaluates one example per request. It is designed for guardrails,
classifiers, policy engines, and gateways that expose a binary prompt-injection decision.

## Request

The default method is `POST` with `Content-Type: application/json`.

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

Change the input field with `--http-text-field prompt` when an API expects `prompt`
instead of `text`.

## Response

A response may provide a label, a score, or both:

```json
{
  "result": {
    "label": "benign",
    "score": 0.14
  }
}
```

Use nested paths when needed:

```bash
--http-label-field result.label \
--http-score-field result.score
```

Scores must be attack probabilities between `0` and `1`. A reverse score such as
`probability_safe` must be converted by an API wrapper or Python predictor.

Accepted positive labels include `attack`, `malicious`, `unsafe`, `blocked`,
`prompt_injection`, `true`, and `1`. Accepted negative labels include `benign`, `safe`,
`allowed`, `false`, and `0`.

## Authentication

Bearer tokens should come from an environment variable so they do not appear directly in
the command:

```bash
export GUARDRAIL_TOKEN='replace-me'
uv run llmst evaluate ... --http-token-env GUARDRAIL_TOKEN
```

Additional headers may be repeated:

```bash
--http-header X-Tenant=security-lab \
--http-header X-Model-Version=2026-07-11
```

Do not put secrets in `--http-header`; shell history and process inspection can expose
them.

## Reliability behavior

- Default timeout: 15 seconds per request.
- Default retries: one retry after the initial request.
- Backoff: bounded exponential delay.
- Default concurrency: four workers.
- Any missing, malformed, or failed response fails the full evaluation by default.

Use `--allow-missing` only for an explicitly partial run. Coverage and excluded IDs are
recorded in `report.json`.

## Data handling

The full input text is sent to the configured endpoint. Report files do not retain that
text, but endpoint logs and upstream infrastructure may. Do not send private production
content unless the endpoint and its logging policy are approved for that data.
