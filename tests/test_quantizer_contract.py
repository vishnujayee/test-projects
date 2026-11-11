"""The contract every quantizer must honour (hard rule 6).

shape, dtype, determinism with a seed, recall above a sanity threshold on the toy
dataset, and "bytes_per_vector matches what is really stored".
"""

from collections.abc import Callable

import numpy as np
import pytest

from trivec.data import make_toy
from trivec.index import exact_search
from trivec.quantizers import (
    BinaryQuantizer,
    Float32Quantizer,
    Int8Quantizer,
    Quantizer,
    TernaryQuantizer,
    make_quantizer,
)

# (factory, minimum recall@10 on the toy dataset).
# Thresholds are deliberately below measured values; they catch breakage, not tune results.
CASES: list[tuple[Callable[[], Quantizer], float]] = [
    (Float32Quantizer, 1.0),
    (Int8Quantizer, 0.95),
    (BinaryQuantizer, 0.30),  # measured 0.418 on 2026-11-01
    (TernaryQuantizer, 0.45),  # measured 0.574 on 2026-11-01
]
IDS = [c[0]().name for c in CASES]


def recall_at_k(found: np.ndarray, truth: np.ndarray) -> float:
    hits = [len(set(f) & set(t)) for f, t in zip(found, truth, strict=True)]
    return float(np.mean(hits) / truth.shape[1])


@pytest.mark.parametrize(("factory", "min_recall"), CASES, ids=IDS)
def test_protocol_and_shapes(factory: Callable[[], Quantizer], min_recall: float) -> None:
    toy = make_toy(n_corpus=200, dim=100)  # 100 is not a multiple of 64: exercises padding
    q = factory()
    assert isinstance(q, Quantizer)
    q.fit(toy.corpus, seed=0)
    codes = q.encode(toy.corpus)
    assert codes.shape[0] == 200
    decoded = q.decode(codes)
    assert decoded.shape == (200, 100)
    assert decoded.dtype == np.float32
    assert np.isfinite(decoded).all()


@pytest.mark.parametrize(("factory", "min_recall"), CASES, ids=IDS)
def test_bytes_per_vector_matches_storage(
    factory: Callable[[], Quantizer], min_recall: float
) -> None:
    for dim in (64, 100, 384):
        toy = make_toy(n_corpus=50, dim=dim)
        q = factory()
        q.fit(toy.corpus, seed=0)
        codes = q.encode(toy.corpus)
        assert codes.nbytes / 50 == q.bytes_per_vector(dim), f"dim={dim}"


@pytest.mark.parametrize(("factory", "min_recall"), CASES, ids=IDS)
def test_deterministic_with_seed(factory: Callable[[], Quantizer], min_recall: float) -> None:
    toy = make_toy(n_corpus=100, dim=64)
    a, b = factory(), factory()
    a.fit(toy.corpus, seed=42)
    b.fit(toy.corpus, seed=42)
    assert np.array_equal(a.encode(toy.corpus), b.encode(toy.corpus))


@pytest.mark.parametrize(("factory", "min_recall"), CASES, ids=IDS)
def test_recall_sanity(factory: Callable[[], Quantizer], min_recall: float) -> None:
    toy = make_toy(n_corpus=1000, n_queries=100, dim=128, k=10)
    q = factory()
    q.fit(toy.corpus, seed=0)
    approx = q.decode(q.encode(toy.corpus))
    ids, _ = exact_search(toy.queries, approx, k=10)
    r = recall_at_k(ids, toy.ground_truth)
    assert r >= min_recall, f"{q.name}: recall@10={r:.3f} < {min_recall}"
    assert r <= 1.0


def test_quality_ordering_on_toy() -> None:
    """More bits should not give worse recall. If it does, something is broken."""
    toy = make_toy(n_corpus=1000, n_queries=100, dim=128, k=10)
    recalls: dict[str, float] = {}
    for factory, _ in CASES:
        q = factory()
        q.fit(toy.corpus, seed=0)
        ids, _ = exact_search(toy.queries, q.decode(q.encode(toy.corpus)), k=10)
        recalls[q.name] = recall_at_k(ids, toy.ground_truth)
    assert recalls["float32"] >= recalls["int8"] >= recalls["ternary"]
    assert recalls["float32"] >= recalls["binary"]


# --- method-specific checks -------------------------------------------------------------


def test_int8_codes_in_range() -> None:
    toy = make_toy(n_corpus=100, dim=32)
    q = Int8Quantizer()
    q.fit(toy.corpus, seed=0)
    codes = q.encode(toy.corpus)
    assert codes.dtype == np.int8
    assert codes.min() >= -127 and codes.max() <= 127


def test_int8_requires_fit() -> None:
    with pytest.raises(RuntimeError):
        Int8Quantizer().encode(make_toy(n_corpus=5, dim=8).corpus)


def test_binary_decode_is_plus_minus_one() -> None:
    toy = make_toy(n_corpus=20, dim=70)
    q = BinaryQuantizer()
    q.fit(toy.corpus, seed=0)
    dec = q.decode(q.encode(toy.corpus))
    assert set(np.unique(dec)) == {-1.0, 1.0}
    assert np.array_equal(dec > 0, toy.corpus > 0)


def test_ternary_values_and_zero_fraction() -> None:
    toy = make_toy(n_corpus=200, dim=128)
    low, high = TernaryQuantizer(alpha=0.3), TernaryQuantizer(alpha=1.2)
    for q in (low, high):
        q.fit(toy.corpus, seed=0)
    codes_low, _ = low.ternarize(toy.corpus)
    codes_high, _ = high.ternarize(toy.corpus)
    assert set(np.unique(codes_low)) <= {-1, 0, 1}
    zeros_low = float(np.mean(codes_low == 0))
    zeros_high = float(np.mean(codes_high == 0))
    assert zeros_low < zeros_high, "a higher alpha must zero out more values"


def test_ternary_decode_matches_definition() -> None:
    """Check the worked example in docs/design.md section 3 by hand."""
    x = np.array([[0.41, -0.02, 0.09, -0.37, 0.01, 0.22, -0.15, 0.00]], dtype=np.float32)
    q = TernaryQuantizer(alpha=0.7)
    q.fit(x, seed=0)
    codes, scale = q.ternarize(x)
    assert codes.tolist() == [[1, 0, 0, -1, 0, 1, -1, 0]]
    assert abs(float(scale[0]) - 0.2875) < 1e-6
    dec = q.decode(q.encode(x))
    np.testing.assert_allclose(dec, codes.astype(np.float32) * 0.2875, atol=1e-6)


def test_ternary_all_zero_vector_is_safe() -> None:
    x = np.zeros((3, 16), dtype=np.float32)
    q = TernaryQuantizer()
    q.fit(x, seed=0)
    dec = q.decode(q.encode(x))
    assert np.all(dec == 0) and np.isfinite(dec).all()


def test_registry_builds_from_config_names() -> None:
    q = make_quantizer("ternary", alpha=0.5)
    assert isinstance(q, TernaryQuantizer) and q.alpha == 0.5
    with pytest.raises(ValueError):
        make_quantizer("nope")
