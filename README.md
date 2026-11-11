# TriVec Lab

An open-source benchmark and agentic research lab for extreme vector compression,
starting with ternary (-1, 0, +1) embeddings for retrieval.

Goal: honest, reproducible numbers on recall vs memory vs speed, and an agent loop
that proposes and tests new quantization variants against a frozen evaluator.

- How it works: [docs/design.md](docs/design.md)
- Results (generated from `results/` only): [docs/results.md](docs/results.md)
- Notebook of what we tried: [docs/lab-notebook.md](docs/lab-notebook.md)

## Quick start

```bash
uv sync --all-groups
uv run pytest -q
```

Status: Phase 0 (skeleton). No benchmark results yet.
