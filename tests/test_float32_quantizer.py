"""Tests for the reference quantizer. Hard rule 6: shape, dtype, determinism, recall."""

import numpy as np

from trivec.data import make_toy
from trivec.index import exact_search
from trivec.quantizers import Float32Quantizer, Quantizer


def test_implements_quantizer_protocol() -> None:
    assert isinstance(Float32Quantizer(), Quantizer)
    assert Float32Quantizer().name == "float32"


def test_encode_shape_and_dtype() -> None:
    toy = make_toy(n_corpus=100, dim=16)
    q = Float32Quantizer()
    q.fit(toy.corpus, seed=0)
    codes = q.encode(toy.corpus)
    assert codes.shape == (100, 16)
    assert codes.dtype == np.float32


def test_deterministic_with_seed() -> None:
    toy = make_toy(n_corpus=100, dim=16)
    a, b = Float32Quantizer(), Float32Quantizer()
    a.fit(toy.corpus, seed=123)
    b.fit(toy.corpus, seed=123)
    assert np.array_equal(a.encode(toy.corpus), b.encode(toy.corpus))


def test_decode_is_exact_roundtrip() -> None:
    toy = make_toy(n_corpus=100, dim=16)
    q = Float32Quantizer()
    assert np.array_equal(q.decode(q.encode(toy.corpus)), toy.corpus)


def test_bytes_per_vector() -> None:
    assert Float32Quantizer().bytes_per_vector(384) == 1536.0


def test_recall_is_perfect() -> None:
    """Searching the decoded vectors must give exactly the ground truth."""
    toy = make_toy()
    q = Float32Quantizer()
    q.fit(toy.corpus, seed=0)
    ids, _ = exact_search(toy.queries, q.decode(q.encode(toy.corpus)), k=10)
    assert np.array_equal(ids, toy.ground_truth)
