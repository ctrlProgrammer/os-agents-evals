# os-agents-evals

CLI that runs the same set of prompts against several models and compares them on accuracy, cost and latency.

## Usage

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/os-evals run examples/dataset.jsonl --models fake-good,fake-cheap --config examples/models.json
```

```text
fake-good: 75% exactitud, $0.48/1k consultas, p95 1.20s
fake-cheap: 25% exactitud, $0.13/1k consultas, p95 0.40s
```

- **Dataset**: JSONL, one `{"prompt": ..., "expected": ...}` per line. Scoring is exact match, ignoring case and surrounding spaces.
- **Models** (`--config`, default `models.json`): name → spec with token prices per million tokens. Only the deterministic `fake` type exists for now.
- **Cost** is the mean cost per query × 1000. **Latency** is p95 (nearest rank).

## Tests

```bash
.venv/bin/pytest
```
