"""The fast path must agree with the slow path (docs/design.md section 9, first trap)."""

import numpy as np
import pytest

from trivec.data import make_toy
from trivec.index import FlatIndex, PackedPopcountIndex, exact_search
from trivec.quantizers import BinaryQuantizer, Float32Quantizer, TernaryQuantizer


def test_flat_index_float32_equals_exact_search() -> None:
    toy = make_toy(n_corpus=500, n_queries=40, dim=96)
    q = Float32Quantizer()
    q.fit(toy.corpus, seed=0)
    idx = FlatIndex(q, q.encode(toy.corpus))
    ids, _scores = idx.search(toy.queries, k=10)
    assert np.array_equal(ids, toy.ground_truth)
    assert len(idx) == 500
    assert idx.mode == "asymmetric_flat"


@pytest.mark.parametrize("dim", [64, 100, 384])
@pytest.mark.parametrize("factory", [BinaryQuantizer, lambda: TernaryQuantizer(alpha=0.7)])
def test_packed_equals_symmetric_slow_path(factory, dim: int) -> None:  # type: ignore[no-untyped-def]
    """Symmetric slow path: decode(encode(query)) against decode(encode(corpus))."""
    toy = make_toy(n_corpus=400, n_queries=30, dim=dim, k=10)
    q = factory()
    q.fit(toy.corpus, seed=0)
    corpus_codes = q.encode(toy.corpus)

    fast = PackedPopcountIndex(q, corpus_codes)
    fast_ids, fast_scores = fast.search(toy.queries, k=10)

    slow_scores_full = q.decode(q.encode(toy.queries)) @ q.decode(corpus_codes).T
    slow_ids, _ = exact_search(q.decode(q.encode(toy.queries)), q.decode(corpus_codes), k=10)

    for i in range(30):
        # Same set of neighbours, allowing for ties at the boundary: every fast id must
        # score at least as high (in the slow metric) as the slow path's 10th result.
        kth = slow_scores_full[i, slow_ids[i, -1]]
        assert np.all(slow_scores_full[i, fast_ids[i]] >= kth - 1e-4), f"query {i}"
        # Scores are proportional: fast/slow ratio is a positive per-query constant.
        slow_sel = slow_scores_full[i, fast_ids[i]]
        nz = np.abs(slow_sel) > 1e-6
        if nz.sum() >= 2:
            ratio = fast_scores[i][nz] / slow_sel[nz]
            assert ratio.min() > 0
            np.testing.assert_allclose(ratio, ratio[0], rtol=1e-3)
    assert fast.mode == "symmetric_packed"


def test_packed_binary_score_is_dim_minus_two_hamming() -> None:
    x = np.array([[1, -1, 1, 1, -1, -1, 1, -1] * 8], dtype=np.float32)  # 64 dims
    y = x.copy()
    y[0, :5] *= -1  # flip 5 bits -> hamming 5
    q = BinaryQuantizer()
    q.fit(x, seed=0)
    idx = PackedPopcountIndex(q, q.encode(np.concatenate([x, y])))
    ids, scores = idx.search(x, k=2)
    assert ids[0, 0] == 0 and scores[0, 0] == 64.0
    assert ids[0, 1] == 1 and scores[0, 1] == 64.0 - 2 * 5


def test_packed_ternary_hand_example() -> None:
    q = TernaryQuantizer(alpha=0.0)  # alpha=0 keeps every non-zero value: pure sign codes
    c = np.array([[1, -1, 0, 1], [1, 1, 1, 1], [-1, 1, 0, -1]], dtype=np.float32)
    q.fit(c, seed=0)
    idx = PackedPopcountIndex(q, q.encode(c))
    query = np.array([[1, -1, 0, 1]], dtype=np.float32)
    ids, scores = idx.search(query, k=3)
    # raw agreements: c0 -> 3, c1 -> 1-1+1 = 1, c2 -> -3 ; each times the corpus scale (all 1.0)
    assert ids[0].tolist() == [0, 1, 2]
    assert scores[0].tolist() == [3.0, 1.0, -3.0]
