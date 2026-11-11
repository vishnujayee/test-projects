"""Tests for the toy dataset and the exact (ground-truth) search."""

import numpy as np
import pytest

from trivec.data import l2_normalize, make_toy
from trivec.index import exact_search


def test_toy_shapes_and_normalisation() -> None:
    toy = make_toy(n_corpus=200, n_queries=20, dim=32, k=5)
    assert toy.corpus.shape == (200, 32)
    assert toy.queries.shape == (20, 32)
    assert toy.source_ids.shape == (20,)
    assert toy.ground_truth.shape == (20, 5)
    assert toy.corpus.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(toy.corpus, axis=1), 1.0, atol=1e-5)
    np.testing.assert_allclose(np.linalg.norm(toy.queries, axis=1), 1.0, atol=1e-5)


def test_toy_is_deterministic() -> None:
    a, b = make_toy(seed=7), make_toy(seed=7)
    assert np.array_equal(a.corpus, b.corpus)
    assert np.array_equal(a.queries, b.queries)
    assert make_toy(seed=8).corpus[0, 0] != a.corpus[0, 0]


def test_planted_neighbour_is_found() -> None:
    """Each query was built from one corpus vector; exact search must rank it first."""
    toy = make_toy()
    top1 = toy.ground_truth[:, 0]
    hit_rate = float(np.mean(top1 == toy.source_ids))
    assert hit_rate >= 0.95, f"planted neighbour found first for only {hit_rate:.0%} of queries"


def test_exact_search_matches_full_sort() -> None:
    rng = np.random.default_rng(0)
    corpus = l2_normalize(rng.standard_normal((50, 8), dtype=np.float32))
    queries = l2_normalize(rng.standard_normal((5, 8), dtype=np.float32))
    ids, scores = exact_search(queries, corpus, k=7)

    expected = np.argsort(-(queries @ corpus.T), axis=1)[:, :7]
    assert np.array_equal(ids, expected)
    assert ids.dtype == np.int64
    assert scores.dtype == np.float32
    assert np.all(np.diff(scores, axis=1) <= 0), "scores must be sorted best-first"


def test_exact_search_k_larger_than_corpus_is_clamped() -> None:
    toy = make_toy(n_corpus=5, n_queries=2, dim=8, k=3)
    ids, _ = exact_search(toy.queries, toy.corpus, k=50)
    assert ids.shape == (2, 5)


def test_exact_search_rejects_bad_k() -> None:
    toy = make_toy(n_corpus=5, n_queries=2, dim=8, k=3)
    with pytest.raises(ValueError):
        exact_search(toy.queries, toy.corpus, k=0)
