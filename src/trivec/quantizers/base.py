"""The Quantizer interface every compression method must implement.

See docs/design.md section 6.1. The key rules:
- `fit` sees ONLY the corpus, never queries (no leakage).
- `encode` and `decode` must be deterministic.
- `decode` exists on every method so one slow, obviously-correct search works for all.
"""

from typing import Any, Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float32]
"""A float32 matrix, shape [n, d]."""

IntArray = npt.NDArray[np.int64]
"""Integer ids, e.g. top-k neighbour indices, shape [n, k]."""

Codes = npt.NDArray[Any]
"""Compressed codes. Each quantizer chooses its own dtype and shape."""


@runtime_checkable
class Quantizer(Protocol):
    """Every compression method implements these four methods."""

    name: str

    def fit(self, corpus: FloatArray, *, seed: int) -> None:
        """Learn anything needed (thresholds, codebooks). Deterministic for a given seed.

        Sees ONLY the corpus. May be a no-op (binary needs nothing).
        """

    def encode(self, x: FloatArray) -> Codes:
        """float32 [n, d] -> compressed codes. Deterministic."""

    def decode(self, codes: Codes) -> FloatArray:
        """codes -> approximate float32 [n, d]. Used by the universal slow search."""

    def bytes_per_vector(self, dim: int) -> float:
        """Memory of ONE encoded vector, including scales and any extra numbers."""
