# Learning

Concepts I learned, written **in my own words** after Claude explains them.
Blank slots are for me to fill in. Links are for reading more.

---

## Concepts

### Embeddings and nearest-neighbour search
- *Claude's version:* a text becomes a list of `d` numbers; similar texts get similar
  lists; search = find the stored lists closest to the query's list (dot product).
- *My words:* …
- Read: https://sbert.net/ (sentence-transformers docs, "Semantic Search")

### Quantization (float32 → int8 → binary → ternary)
- *Claude's version:* trade precision for memory; each level keeps fewer bits per
  number; ternary keeps the sign plus an explicit "this one is small, drop it".
- *My words:* …
- Read: https://huggingface.co/blog/embedding-quantization (binary and int8 embeddings, 2024)

### Recall@k vs nDCG@k
- *Claude's version:* recall = agreement with exact float32 search; nDCG = agreement
  with human relevance labels, position-weighted.
- *My words:* …
- Read: BEIR benchmark paper — https://arxiv.org/abs/2104.08663

### Asymmetric vs symmetric distance
- *Claude's version:* asymmetric keeps the query precise and only compresses the
  database; symmetric compresses both and unlocks bit-tricks.
- *My words:* …

### Product quantization (PQ)
- *Claude's version:* cut the vector into `m` chunks, replace each chunk with the ID
  of its nearest learned "typical chunk"; store `m` small IDs instead of `d` floats.
- *My words:* …
- Read: Jégou et al. 2011 — https://inria.hal.science/inria-00514462/document ;
  FAISS library paper — https://arxiv.org/abs/2401.08281

### Popcount search
- *Claude's version:* pack bits into 64-bit words; XOR/AND then count 1-bits with one
  CPU instruction; ternary needs four popcounts per word.
- *My words:* …

### Why rotations help extreme quantization
- *Claude's version:* a random orthogonal rotation spreads information evenly over
  dimensions so a fixed threshold loses less.
- *My words:* …
- Read: RaBitQ — https://arxiv.org/abs/2405.12497

### Ternary weights in neural nets (related idea)
- *Claude's version:* BitNet b1.58 showed `{-1, 0, +1}` weights can match full
  precision in LLMs; we ask whether the same holds for *embeddings*.
- *My words:* …
- Read: https://arxiv.org/abs/2402.17764

### Matryoshka embeddings (related idea)
- *Claude's version:* models trained so the first `n` dimensions are already a good
  embedding; lets you truncate before compressing.
- *My words:* …
- Read: https://arxiv.org/abs/2205.13147

### Top-k with argpartition
- *Claude's version:* `np.argpartition(-scores, k-1)` moves the k largest scores to
  the front without ordering the rest; then we sort only those k. Same result as a
  full sort, far less work when the corpus is large.
- *My words:* …
- Read: https://numpy.org/doc/stable/reference/generated/numpy.argpartition.html

### Protocols (structural typing)
- *Claude's version:* a `Protocol` says "anything with these methods counts". A class
  does not need to inherit from it; it just has to have the right methods. mypy checks
  this, and `@runtime_checkable` lets `isinstance` check it too.
- *My words:* …
- Read: https://typing.python.org/en/latest/spec/protocol.html

### Reproducibility and provenance
- *Claude's version:* a result is only real if someone else can re-run it; so every
  result file records config, seed, code version, library versions, hardware, data
  and model versions.
- *My words:* …

### Frozen evaluator (and why hashes, not promises)
- *Claude's version:* the judge must not change while the contest runs; a hash file
  plus a CI test makes any change fail loudly.
- *My words:* …

---

## Tools

- **uv** — fast Python package/project manager. Read: https://docs.astral.sh/uv/
- **ruff** — linter + formatter. Read: https://docs.astral.sh/ruff/
- **mypy strict** — type checker; catches wrong shapes of code before running.
- **pytest** — tests. Read: https://docs.pytest.org/
- **FAISS** — Meta's similarity-search library; we use its PQ. Read: https://github.com/facebookresearch/faiss/wiki
- **Ollama** — run open LLMs locally. Read: https://ollama.com/
