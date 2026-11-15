"""Contract and method-specific tests for the multi-view (rotated / residual) ternary quantizer."""

import numpy as np
import pytest

from trivec.data import make_toy
from trivec.index import exact_search
from trivec.quantizers import Quantizer, TernaryQuantizer, make_quantizer
from trivec.quantizers.rternary import RotatedTernaryQuantizer

VARIANTS = [
    {"rotate": False, "residual": False},
    {"rotate": True, "residual": False},
    {"rotate": False, "residual": True},
    {"rotate": True, "residual": True},
]
IDS = [f"rot={v['rotate']}-res={v['residual']}" for v in VARIANTS]


def recall_at_k(found: np.ndarray, truth: np.ndarray) -> float:
    hits = [len(set(f) & set(t)) for f, t in zip(found, truth, strict=True)]
    return float(np.mean(hits) / truth.shape[1])


def toy_recall(q: Quantizer) -> float:
    toy = make_toy(n_corpus=1000, n_queries=100, dim=128, k=10)
    q.fit(toy.corpus, seed=0)
    ids, _ = exact_search(toy.queries, q.decode(q.encode(toy.corpus)), k=10)
    return recall_at_k(ids, toy.ground_truth)


@pytest.mark.parametrize("params", VARIANTS, ids=IDS)
def test_protocol_shapes_and_finite(params: dict[str, bool]) -> None:
    toy = make_toy(n_corpus=200, dim=100)  # 100 is not a multiple of 64: exercises padding
    q = RotatedTernaryQuantizer(**params)
    assert isinstance(q, Quantizer)
    q.fit(toy.corpus, seed=0)
    decoded = q.decode(q.encode(toy.corpus))
    assert decoded.shape == (200, 100)
    assert decoded.dtype == np.float32
    assert np.isfinite(decoded).all()


@pytest.mark.parametrize("params", VARIANTS, ids=IDS)
def test_bytes_per_vector_matches_storage(params: dict[str, bool]) -> None:
    for dim in (64, 100, 384):
        toy = make_toy(n_corpus=50, dim=dim)
        q = RotatedTernaryQuantizer(**params)
        q.fit(toy.corpus, seed=0)
        assert q.encode(toy.corpus).nbytes / 50 == q.bytes_per_vector(dim), f"dim={dim}"


@pytest.mark.parametrize("params", VARIANTS, ids=IDS)
def test_deterministic_with_seed(params: dict[str, bool]) -> None:
    toy = make_toy(n_corpus=100, dim=64)
    a, b = RotatedTernaryQuantizer(**params), RotatedTernaryQuantizer(**params)
    a.fit(toy.corpus, seed=42)
    b.fit(toy.corpus, seed=42)
    assert np.array_equal(a.encode(toy.corpus), b.encode(toy.corpus))


def test_rotation_is_orthogonal() -> None:
    q = RotatedTernaryQuantizer(rotate=True)
    q.fit(np.zeros((10, 96), dtype=np.float32), seed=0)
    r = q._rotation
    assert r is not None
    assert np.allclose(r.T @ r, np.eye(96), atol=1e-5)


def test_no_rotation_no_residual_equals_plain_ternary() -> None:
    """With both knobs off, the result must match plain ternary exactly."""
    toy = make_toy(n_corpus=200, dim=128)
    plain = TernaryQuantizer(alpha=0.7)
    multi = RotatedTernaryQuantizer(alpha=0.7, rotate=False, residual=False)
    plain.fit(toy.corpus, seed=0)
    multi.fit(toy.corpus, seed=0)
    assert np.array_equal(plain.encode(toy.corpus), multi.encode(toy.corpus))


def test_residual_reduces_reconstruction_error() -> None:
    toy = make_toy(n_corpus=500, dim=128)
    one = RotatedTernaryQuantizer(rotate=True, residual=False)
    two = RotatedTernaryQuantizer(rotate=True, residual=True)
    one.fit(toy.corpus, seed=0)
    two.fit(toy.corpus, seed=0)
    err1 = np.linalg.norm(toy.corpus - one.decode(one.encode(toy.corpus)))
    err2 = np.linalg.norm(toy.corpus - two.decode(two.encode(toy.corpus)))
    assert err2 < err1


def test_recall_sanity_and_registry() -> None:
    q = make_quantizer("rternary", alpha=0.7, rotate=True, residual=True)
    assert toy_recall(q) >= 0.45  # catches breakage, does not tune results
