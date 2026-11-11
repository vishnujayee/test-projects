"""A small synthetic dataset for tests (hard rule 6).

Corpus vectors are random. Each query is a copy of one corpus vector plus a little
noise, so we know which document it "should" find. That makes a recall sanity
threshold meaningful without downloading anything.
"""

from dataclasses import dataclass

import numpy as np

from trivec.index.exact import exact_search
from trivec.quantizers.base import FloatArray, IntArray


def l2_normalize(x: FloatArray) -> FloatArray:
    """Scale every row to length 1 so dot product equals cosine similarity."""
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return np.ascontiguousarray(x / norms, dtype=np.float32)


@dataclass(frozen=True)
class ToyDataset:
    corpus: FloatArray
    """[n_corpus, dim], L2-normalised."""
    queries: FloatArray
    """[n_queries, dim], L2-normalised."""
    source_ids: IntArray
    """[n_queries]: which corpus row each query was built from."""
    ground_truth: IntArray
    """[n_queries, k]: exact float32 top-k ids (hard rule 4)."""


def make_toy(
    n_corpus: int = 1000,
    n_queries: int = 50,
    dim: int = 64,
    *,
    noise: float = 0.1,
    k: int = 10,
    seed: int = 0,
) -> ToyDataset:
    """Build a deterministic toy dataset. Same seed -> identical arrays.

    n_queries is clamped to n_corpus because each query is built from a distinct document.
    """
    n_queries = min(n_queries, n_corpus)
    rng = np.random.default_rng(seed)
    corpus = l2_normalize(rng.standard_normal((n_corpus, dim), dtype=np.float32))

    source_ids = rng.choice(n_corpus, size=n_queries, replace=False).astype(np.int64)
    jitter = noise * rng.standard_normal((n_queries, dim), dtype=np.float32)
    queries = l2_normalize(corpus[source_ids] + jitter)

    ground_truth, _ = exact_search(queries, corpus, k)
    return ToyDataset(corpus, queries, source_ids, ground_truth)
