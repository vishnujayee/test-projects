"""int8 scalar quantization: 1 byte per dimension.

`fit` learns one scale per DIMENSION from the corpus (the largest |value| seen in
that dimension). Each value is then rounded to an integer in [-127, 127].
This per-dimension calibration is the standard, strong form of int8 (not a straw man).

Shared parameters (the d scales) are not counted in bytes_per_vector: they are O(d)
and do not grow with the corpus.
"""

import numpy as np

from trivec.quantizers.base import Codes, FloatArray


class Int8Quantizer:
    name = "int8"

    def __init__(self) -> None:
        self._scale: FloatArray | None = None  # [d], value that maps to 127

    def fit(self, corpus: FloatArray, *, seed: int) -> None:
        absmax = np.abs(corpus).max(axis=0).astype(np.float32)
        absmax[absmax == 0] = 1.0  # a constant-zero dimension must not divide by zero
        self._scale = absmax / 127.0

    def encode(self, x: FloatArray) -> Codes:
        if self._scale is None:
            raise RuntimeError("call fit() before encode()")
        q = np.rint(x / self._scale)
        return np.clip(q, -127, 127).astype(np.int8)

    def decode(self, codes: Codes) -> FloatArray:
        if self._scale is None:
            raise RuntimeError("call fit() before decode()")
        return np.ascontiguousarray(codes.astype(np.float32) * self._scale, dtype=np.float32)

    def bytes_per_vector(self, dim: int) -> float:
        return float(dim)
