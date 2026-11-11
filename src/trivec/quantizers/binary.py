"""Binary quantization: 1 bit per dimension, the sign of each value.

Nothing to learn. Codes are a packed bit-plane (1 = positive). `decode` returns
+1/-1 per dimension; since every vector then has the same length, the missing
scale does not change any ranking.
"""

import numpy as np

from trivec.quantizers.base import Codes, FloatArray
from trivec.quantizers.bits import pack_plane, plane_bytes, unpack_plane


class BinaryQuantizer:
    name = "binary"

    def __init__(self) -> None:
        self._dim: int | None = None

    def fit(self, corpus: FloatArray, *, seed: int) -> None:
        self._dim = int(corpus.shape[1])

    def encode(self, x: FloatArray) -> Codes:
        self._dim = int(x.shape[1])
        return pack_plane(x > 0)

    def decode(self, codes: Codes) -> FloatArray:
        if self._dim is None:
            raise RuntimeError("call fit() or encode() before decode()")
        bits = unpack_plane(codes, self._dim)
        return np.where(bits, np.float32(1.0), np.float32(-1.0)).astype(np.float32)

    def bytes_per_vector(self, dim: int) -> float:
        return float(plane_bytes(dim))
