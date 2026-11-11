# Lab Notebook

Dated entries. Newest at the bottom. Each entry: what we did, what I learned,
what went wrong, what is next. Failed experiments go here too (hard rule 9).

---

## 2026-11-01 — Session 1: project docs and first design

**What we did**
- Created `docs/` with all six project documents.
- Wrote the first full version of `design.md`: the one-paragraph story, a glossary,
  a worked ternary example, the two halves (Bench, Lab), module interfaces,
  the frozen-evaluator mechanism, the Lab state machine and sandbox, a decisions
  table, a traps-and-guards table, and a 10-step milestone list.
- Took seven design decisions on my behalf and listed them in `design.md` §12 for
  me to confirm or change.

**What I learned** (fill in your own words after reading; see `learning.md`)
- Recall@10 vs nDCG@10 measure different things.
- Asymmetric vs symmetric search, and why mixing them up misleads.
- Why `decode()` on every quantizer gives one universal, slow, correct search.
- "Frozen" is enforced with a hash file and a CI test, not just a rule.

**What went wrong**
- Nothing yet. No code exists. The folder is not a git repo yet.

**Next**
- Confirm or change the decisions in `design.md` §12.
- Task 2: skeleton (`git init`, `pyproject.toml` with uv, ruff, mypy, pytest,
  GitHub Actions, `Quantizer` protocol, `float32` quantizer, toy dataset, one test).

---

## 2026-11-01 — Session 1 (continued): Milestone 1, the skeleton

**What we did**
- Confirmed the seven design decisions. Installed `uv` (it was not on the machine).
- `git init` on branch `main`; `.gitignore` excludes `.venv`, caches, `data/cache/`, `.env`.
- `pyproject.toml`: Python ≥ 3.11 (pinned 3.12 via `.python-version`), deps numpy /
  pyyaml / typer, dev group pytest / pytest-cov / ruff / mypy. Ruff line length 100,
  mypy strict on `src/`.
- Code: `quantizers/base.py` (the `Quantizer` protocol + `FloatArray`, `IntArray`,
  `Codes` type aliases), `quantizers/float32.py` (identity reference),
  `index/exact.py` (`exact_search`, the ground-truth function, hard rule 4),
  `data/toy.py` (`make_toy`, `l2_normalize`), minimal `cli.py` with `trivec info`.
- Tests: 12, covering shape, dtype, determinism, exact roundtrip, bytes/vector,
  perfect recall for float32; toy shapes/normalisation/determinism, planted
  neighbour found first, exact search equals a full sort, k clamping, bad k rejected.
- CI: `.github/workflows/ci.yml` runs ruff check, ruff format --check, mypy, pytest.
- All green locally: ruff, format, mypy (9 files), pytest (12 passed), `trivec info`.

**What I learned**
- `np.argpartition` finds the top-k without sorting the whole row; we then sort only
  those k. Same answer as a full sort, much faster on big corpora.
- Typer treats a single command as the root command; an `@app.callback()` keeps
  subcommands as subcommands.
- A `runtime_checkable` Protocol lets `isinstance(obj, Quantizer)` work in tests.

**What went wrong**
- `uv` was not installed; installed with the official installer to `~/.local/bin`
  (not yet on PATH for new shells until restart).
- Heavy deps (faiss-cpu, sentence-transformers, datasets) are deliberately NOT added
  yet. sentence-transformers pulls in torch; we will add it in Milestone 4 with a
  CPU-only torch index so the install stays small.

**Next**
- Commit this skeleton (nothing committed yet).
- Milestone 2: ternary, binary and int8 quantizers with the four required tests each.
  CORE code: line by line, you write one part.
