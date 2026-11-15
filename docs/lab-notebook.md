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

---

## 2026-11-15 — Multi-view ternary: rotation + residual (claims C8–C11)

**What was tried**
- New quantizer `rternary` (`src/trivec/quantizers/rternary.py`), two optional "views" on top
  of plain ternary: (1) a random orthogonal rotation `R` (384×384, drawn once in `fit()` from the
  seed) applied before ternarizing; (2) a residual ternary code that stores the error of the first
  code. Both knobs ablated: off/off, rotate only, residual only, both. alpha = 0.7 throughout.
- Claims C8–C11 were appended to `configs/claims.yaml` and committed (85f8eb1) before any
  benchmark run. Claim label strings matched what `trivec report` prints, so no amendment was needed.
- Configs `configs/{scifact,nfcorpus,arguana}_multiview.yaml`: pq(m=192) plus the four rternary
  variants, seeds 0/1/2, asymmetric_flat only. Run in a Linux cloud container (4-core Xeon);
  the older ternary/PQ results came from a Windows laptop. Same package versions.
- The evaluator (`src/trivec/eval/`) was not touched; the frozen-hash test passes.

**Verdicts** (paired bootstrap over queries, 95% interval, all-MiniLM-L6-v2)

| claim | arguana | nfcorpus | scifact | overall |
|---|---|---|---|---|
| C8 rotation beats plain ternary on recall (100 B) | supported, +0.0075 | inconclusive, −0.0003 | inconclusive, +0.0088 | mixed |
| C9 residual beats rotation-only on recall | supported, +0.048 | supported, +0.119 | supported, +0.094 | supported everywhere |
| C10 rot+res beats pq(m=192) on recall | contradicted, −0.040 | contradicted, −0.071 | contradicted, −0.064 | contradicted everywhere |
| C11 rot+res beats pq(m=192) on nDCG | inconclusive, −0.0007 | inconclusive, −0.0023 | contradicted, −0.0094 | not supported |

(The report shows "mixed" for C9/C10 overall only because a `scifact / bge-small-en-v1.5` group
exists in `results/` where these methods were not run; on the three MiniLM datasets the verdicts
are uniform.)

**Cost of the rotation matrix ("with shared state")**
- R is 384 × 384 float32 = 589,824 bytes, shared by all vectors. Amortized over the corpus that
  adds +68 B/vector on arguana (8,674 docs), +114 on scifact (5,183), +162 on nfcorpus (3,633).
- So rotated 100-byte codes cost 168 / 214 / 262 bytes with shared state, and rotated+residual
  200-byte codes cost 268 / 314 / 362 — more than pq(m=192) with its codebooks (237 / 268 / 300).
  On these small corpora rotation is not free; it would only become negligible at ~10^6 vectors.

**Unverified results / what surprised me**
- The residual does nearly all the work. Residual-only (no rotation, 200 B, no shared state)
  gets recall 0.920 / 0.879 / 0.875 vs rotated+residual 0.924 / 0.880 / 0.883. Rotation adds less
  than 0.01 recall at best, and on scifact it *lowers* nDCG (0.643 → 0.629 at 100 B,
  0.653 → 0.640 at 200 B). Rotation on already-dense sentence embeddings seems to buy little;
  this is different from the ITQ / RaBitQ setting I expected it to mirror.
- PQ at equal per-vector memory stays clearly ahead on recall (0.95–0.96 vs 0.88–0.92), the same
  pattern as C3 at 100 bytes. Doubling ternary memory did not close the gap to PQ.
- nDCG is close across 200-byte methods (within ~0.01), so for end users the compressed
  methods are near-indistinguishable here; recall vs exact search is where they separate.
- Sanity check: the off/off variant equals plain ternary exactly on scifact. On nfcorpus and
  arguana it differs by ≤ 0.0003 recall / ≤ 0.0005 nDCG (interval contains 0) — most likely tiny
  embedding differences between the Windows and Linux runs. C8 gives the same verdicts when
  compared against the same-run off/off variant instead (exploratory check, not a claim).
- Environment note: the cloud network needed download.pytorch.org, huggingface.co plus its CDNs
  (us.aws.cdn.hf.co, cas-server.xethub.hf.co) and the BEIR host allowed before the bench could run.

**Next**
- If pursued: residual-only ternary is the cheaper variant worth keeping; drop rotation or test a
  learned rotation (ITQ) instead of a random one. A fair PQ comparison should also report the
  "with shared state" column, where both sides pay for shared parameters.
