# Contributing

Contributions should improve reproducibility, detector compatibility, metrics, or report
clarity without weakening the interpretation boundaries.

## Development setup

```bash
git clone https://github.com/3nesdeniz/llm-security-testbench.git
cd llm-security-testbench
uv sync --dev
uv run ruff check .
uv run mypy src
uv run pytest
uv build
```

## Pull requests

- Keep each pull request focused on one behavior.
- Add or update tests for every code change.
- Do not include API keys, customer data, private prompts, or production logs.
- Document new label formats, endpoint fields, metrics, and CLI flags.
- Preserve raw-text omission in generated reports unless a future opt-in design receives
  a separate security review.
- State any compatibility impact in the pull request description.

## Dataset integrations

New dataset adapters must define the label boundary and licensing expectations. Avoid
hard-coding one dataset's IDs or taxonomy into the evaluation engine.

## Reporting issues

Use GitHub Issues for bugs and feature requests. Report security vulnerabilities through
the private process in [SECURITY.md](SECURITY.md).
