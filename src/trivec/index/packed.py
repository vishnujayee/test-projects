"""PackedPopcountIndex: the fast, symmetric search for binary and ternary codes.

Both query and corpus are encoded. Rows of packed bits are viewed as uint64 words and
compared with XOR/AND + popcount (count of 1-bits, one CPU instruction per word).

  binary : score = d - 2 * popcount(q XOR c)
  ternary: score = [pc(q+ & c+) + pc(q- & c-) - pc(q+ & c-) - pc(q- & c+)] * scale_c

The ternary score is multiplied by the corpus vector's scale so the ranking equals the
slow path (decode both sides, dot product) exactly; the query's own scale is a positive
constant per query and cannot change the order.

Search is one query at a time on purpose: that is what we time for p50/p95 latency.
"""

import numpy as np
import numpy.typing as npt

from trivec.quantizers.base import Codes, FloatArray, IntArray
from trivec.quantizers.binary import BinaryQuantizer
from trivec.quantizers.bits import plane_bytes
from trivec.quantizers.ternary import TernaryQuantizer

Words = npt.NDArray[np.uint64]


def _as_words(packed: Codes) -> Words:
    """uint8 [n, 8w] -> uint64 [n, w]. Rows are padded to whole words by pack_plane."""
    return np.ascontiguousarray(packed).view(np.uint64)


def _popcount(words: Words) -> npt.NDArray[np.int64]:
    return np.bitwise_count(words).sum(axis=1, dtype=np.int64)


class PackedPopcountIndex:
    mode = "symmetric_packed"

    def __init__(self, quantizer: BinaryQuantizer | TernaryQuantizer, corpus_codes: Codes) -> None:
        self.quantizer = quantizer
        self.codes = corpus_codes
        if isinstance(quantizer, TernaryQuantizer):
            pos, neg, scale = quantizer.split(corpus_codes)
            self._pos, self._neg, self._scale = _as_words(pos), _as_words(neg), scale
            self._dim = int(quantizer._dim or 0)
        else:
            self._bits = _as_words(corpus_codes)
            if quantizer._dim is None:
                raise RuntimeError("binary quantizer must be fitted before indexing")
            self._dim = int(quantizer._dim)  # true dim: padding bits are 0 on both sides
        self._n = int(corpus_codes.shape[0])

    def __len__(self) -> int:
        return self._n

    def _scores_one(self, query_code: Codes) -> npt.NDArray[np.float32]:
        q = self.quantizer
        if isinstance(q, TernaryQuantizer):
            qpos, qneg, _ = q.split(query_code)
            qp, qn = _as_words(qpos)[0], _as_words(qneg)[0]
            raw = (
                _popcount(qp & self._pos)
                + _popcount(qn & self._neg)
                - _popcount(qp & self._neg)
                - _popcount(qn & self._pos)
            )
            return (raw.astype(np.float32) * self._scale).astype(np.float32)
        qb = _as_words(query_code)[0]
        hamming = _popcount(qb ^ self._bits)
        return (self._dim - 2 * hamming).astype(np.float32)

    def search(self, queries: FloatArray, k: int) -> tuple[IntArray, FloatArray]:
        codes = self.quantizer.encode(queries)
        k = min(k, self._n)
        ids = np.empty((len(codes), k), dtype=np.int64)
        scores = np.empty((len(codes), k), dtype=np.float32)
        for i in range(len(codes)):
            s = self._scores_one(codes[i : i + 1])
            part = np.argpartition(-s, k - 1)[:k]
            order = np.argsort(-s[part], kind="stable")
            ids[i] = part[order]
            scores[i] = s[part[order]]
        return ids, scores


def plane_words(dim: int) -> int:
    """How many uint64 words one bit-plane of `dim` bits occupies."""
    return plane_bytes(dim) // 8
