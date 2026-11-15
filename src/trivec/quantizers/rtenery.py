"""Multi-view ternary: rotate first, then optionally store a ternary correction.

Idea (Vishnu's "current / alternative / what-if" views of one vector):
  view 1  rotated   y = x @ R, where R is a random orthogonal matrix learned once in fit().
                    Rotation spreads the vector's energy evenly over all dimensions, so a
                    single threshold throws away less signal (same principle as ITQ / RaBitQ).
  view 2  ternary   c1 = ternary(y) with scale s1                 (the "current" code)
  view 3  residual  c2 = ternary(y - s1 * c1) with scale s2       (the "alternative": a
                    correction for what view 2 got wrong)

R is orthogonal, so inner products are preserved: <q, x> = <q R, x R>. decode() maps back with
R^T, which keeps every existing search mode and the frozen evaluator untouched.

Knobs, so each part can be tested on its own (ablation):
  rotate=False, residual=False  -> identical to plain ternary (sanity check)
  rotate=True,  residual=False  -> rotation only, same 100 bytes per vector at d=384
  rotate=False, residual=True   -> residual only, about 200 bytes
  rotate=True,  residual=True   -> both

Memory: per-vector codes are counted in bytes_per_vector. R (d*d float32) is shared state; the
evaluator counts it automatically in "with shared state", so the cost is reported honestly.

Storage per vector (uint8 row): [pos1 | neg1 | s1 (4 bytes)] and, if residual, [pos2 | neg2 | s2].
"""

import numpy as np

from trivec.quantizers.base import Codes, FloatArray
from trivec.quantizers.bits import pack_plane, plane_bytes, unpack_plane
from trivec.quantizers.ternary import TernaryQuantizer


class RotatedTernaryQuantizer:
    name = "rternary"

    def __init__(self, alpha: float = 0.7, rotate: bool = True, residual: bool = False) -> None:
        if alpha < 0:
            raise ValueError("alpha must be >= 0")
        self.alpha = float(alpha)
        self.rotate = bool(rotate)
        self.residual = bool(residual)
        self._tern = TernaryQuantizer(alpha=self.alpha)
        self._rotation: FloatArray | None = None
        self._dim: int | None = None

    # -- fit ------------------------------------------------------------------------------
    def fit(self, corpus: FloatArray, *, seed: int) -> None:
        """Sees only the corpus shape. R is random, so it cannot leak query information."""
        d = int(corpus.shape[1])
        self._dim = d
        if self.rotate:
            rng = np.random.default_rng(seed)
            q, r = np.linalg.qr(rng.standard_normal((d, d)))
            q *= np.sign(np.diag(r))  # makes the QR result unique, so R is deterministic
            self._rotation = np.ascontiguousarray(q, dtype=np.float32)

    # -- helpers --------------------------------------------------------------------------
    def _to_rotated(self, x: FloatArray) -> FloatArray:
        if not self.rotate:
            return np.asarray(x, dtype=np.float32)
        if self._rotation is None:
            raise RuntimeError("call fit() before encode()")
        return np.ascontiguousarray(x @ self._rotation, dtype=np.float32)

    def _from_rotated(self, y: FloatArray) -> FloatArray:
        if not self.rotate:
            return np.ascontiguousarray(y, dtype=np.float32)
        assert self._rotation is not None
        return np.ascontiguousarray(y @ self._rotation.T, dtype=np.float32)

    @staticmethod
    def _pack(codes: Codes, scale: FloatArray) -> Codes:
        pos = pack_plane(codes > 0)
        neg = pack_plane(codes < 0)
        return np.concatenate([pos, neg, scale.reshape(-1, 1).view(np.uint8)], axis=1)

    def _unpack(self, block: Codes) -> FloatArray:
        assert self._dim is not None
        pb = plane_bytes(self._dim)
        pos = unpack_plane(block[:, :pb], self._dim).astype(np.float32)
        neg = unpack_plane(block[:, pb : 2 * pb], self._dim).astype(np.float32)
        scale = np.ascontiguousarray(block[:, 2 * pb : 2 * pb + 4]).view(np.float32).reshape(-1)
        return (pos - neg) * scale[:, None]

    def _block_bytes(self, dim: int) -> int:
        return 2 * plane_bytes(dim) + 4

    # -- protocol -------------------------------------------------------------------------
    def encode(self, x: FloatArray) -> Codes:
        if self._dim is None:
            raise RuntimeError("call fit() before encode()")
        y = self._to_rotated(x)
        c1, s1 = self._tern.ternarize(y)
        blocks = [self._pack(c1, s1)]
        if self.residual:
            r = (y - c1.astype(np.float32) * s1[:, None]).astype(np.float32)
            c2, s2 = self._tern.ternarize(r)
            blocks.append(self._pack(c2, s2))
        return np.concatenate(blocks, axis=1)

    def decode(self, codes: Codes) -> FloatArray:
        if self._dim is None:
            raise RuntimeError("call fit() before decode()")
        bb = self._block_bytes(self._dim)
        y_hat = self._unpack(codes[:, :bb])
        if self.residual:
            y_hat = y_hat + self._unpack(codes[:, bb : 2 * bb])
        return self._from_rotated(y_hat)

    def bytes_per_vector(self, dim: int) -> float:
        return float(self._block_bytes(dim) * (2 if self.residual else 1))
