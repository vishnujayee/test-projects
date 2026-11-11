"""PQ baseline tests (hard rule 6). FAISS-backed, so a little slower than the others."""

import numpy as np
import pytest

from trivec.data import make_toy
from trivec.index import FlatIndex
from trivec.quantizers import Quantizer, make_quantizer
from trivec.quantizers.pq import PQQuantizer

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")  # faiss "few training points"


def _recall(found: np.ndarray, truth: np.ndarray) -> float:
    return float(np.mean([len(set(f) & set(t)) for f, t in zip(found, truth, strict=True)]) / 10)


def test_pq_shape_dtype_bytes() -> None:
    toy = make_toy(n_corpus=600, dim=128)
    q = PQQuantizer(m=16)
    assert isinstance(q, Quantizer)
    q.fit(toy.corpus, seed=0)
    codes = q.encode(toy.corpus)
    assert codes.shape == (600, 16) and codes.dtype == np.uint8
    assert codes.nbytes / 600 == q.bytes_per_vector(128) == 16.0
    dec = q.decode(codes)
    assert dec.shape == (600, 128) and dec.dtype == np.float32


def test_pq_codebook_memory_is_counted() -> None:
    from trivec.eval.memory import shared_state_bytes

    toy = make_toy(n_corpus=600, dim=128)
    q = PQQuantizer(m=16)
    q.fit(toy.corpus, seed=0)
    codebook = 16 * 256 * (128 // 16) * 4  # subvectors x centroids x sub-dimension x float32
    assert codebook <= shared_state_bytes(q) < codebook + 1024


def test_pq_deterministic_with_seed() -> None:
    toy = make_toy(n_corpus=600, dim=64)
    a, b = PQQuantizer(m=8), PQQuantizer(m=8)
    a.fit(toy.corpus, seed=3)
    b.fit(toy.corpus, seed=3)
    assert np.array_equal(a.encode(toy.corpus), b.encode(toy.corpus))


def test_pq_recall_sanity() -> None:
    toy = make_toy(n_corpus=1000, n_queries=100, dim=128, k=10)
    q = PQQuantizer(m=32)  # 32 bytes/vector, i.e. 16x smaller than float32
    q.fit(toy.corpus, seed=0)
    ids, _ = FlatIndex(q, q.encode(toy.corpus)).search(toy.queries, k=10)
    r = _recall(ids, toy.ground_truth)
    assert 0.30 <= r <= 1.0, f"pq recall@10={r:.3f}"


def test_pq_rejects_bad_dims_and_requires_fit() -> None:
    toy = make_toy(n_corpus=50, dim=100)
    with pytest.raises(ValueError):
        PQQuantizer(m=48).fit(toy.corpus, seed=0)
    with pytest.raises(RuntimeError):
        PQQuantizer(m=4).encode(toy.corpus)
    assert isinstance(make_quantizer("pq", m=4), PQQuantizer)
