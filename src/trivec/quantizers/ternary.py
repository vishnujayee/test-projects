"""Ternary quantization: every value becomes -1, 0 or +1 (see docs/design.md section 3).

Per vector:
  threshold t = alpha * mean(|x|)        values with |x| < t become 0
  scale     s = mean(|x|) over non-zeros  used by decode: x_hat = s * code

Storage per vector (one uint8 row): [ pos plane | neg plane | scale as 4 bytes ].
pos plane bit i = 1 if code_i == +1; neg plane bit i = 1 if code_i == -1.
"""

import numpy as np

from trivec.quantizers.base import Codes, FloatArray
from trivec.quantizers.bits import pack_plane, plane_bytes, unpack_plane


class TernaryQuantizer:
    name = "ternary"

    def __init__(self, alpha: float = 0.7) -> None:
        if alpha < 0:
            raise ValueError("alpha must be >= 0")
        self.alpha = float(alpha)
        self._dim: int | None = None

    def fit(self, corpus: FloatArray, *, seed: int) -> None:
        self._dim = int(corpus.shape[1])

    def ternarize(self, x: FloatArray) -> tuple[Codes, FloatArray]:
        """Return (codes in {-1,0,+1} as int8 [n,d], scale per vector [n])."""
        a = np.abs(x)
        threshold = self.alpha * a.mean(axis=1, keepdims=True)
        keep = a >= threshold
        keep &= a > 0  # a value of exactly 0 is never "kept" (matters when alpha == 0)
        codes = np.where(keep, np.sign(x), 0).astype(np.int8)
        n_keep = keep.sum(axis=1)
        kept_sum = np.where(keep, a, 0).sum(axis=1)
        scale = np.divide(kept_sum, n_keep, out=np.zeros_like(kept_sum), where=n_keep > 0)
        return codes, scale.astype(np.float32)

    def encode(self, x: FloatArray) -> Codes:
        self._dim = int(x.shape[1])
        codes, scale = self.ternarize(x)
        pos = pack_plane(codes > 0)
        neg = pack_plane(codes < 0)
        scale_bytes = scale.reshape(-1, 1).view(np.uint8)  # float32 -> 4 bytes each
        return np.concatenate([pos, neg, scale_bytes], axis=1)

    def split(self, codes: Codes) -> tuple[Codes, Codes, FloatArray]:
        """Split packed rows into (pos plane, neg plane, scale)."""
        if self._dim is None:
            raise RuntimeError("call fit() or encode() before decode()")
        pb = plane_bytes(self._dim)
        pos = codes[:, :pb]
        neg = codes[:, pb : 2 * pb]
        scale = np.ascontiguousarray(codes[:, 2 * pb : 2 * pb + 4]).view(np.float32).reshape(-1)
        return pos, neg, scale

    def decode(self, codes: Codes) -> FloatArray:
        pos, neg, scale = self.split(codes)
        assert self._dim is not None
        signs = unpack_plane(pos, self._dim).astype(np.float32) - unpack_plane(
            neg, self._dim
        ).astype(np.float32)
        return np.ascontiguousarray(signs * scale[:, None], dtype=np.float32)

    def bytes_per_vector(self, dim: int) -> float:
        return float(2 * plane_bytes(dim) + 4)
