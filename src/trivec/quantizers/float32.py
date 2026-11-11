"""The reference "quantizer": no compression at all.

Its numbers define perfect recall and the memory ceiling (4 bytes per dimension).
Every other method is compared against this one.
"""

import numpy as np

from trivec.quantizers.base import Codes, FloatArray


class Float32Quantizer:
    """Identity: codes are the original float32 vectors."""

    name = "float32"

    def fit(self, corpus: FloatArray, *, seed: int) -> None:
        """Nothing to learn."""

    def encode(self, x: FloatArray) -> Codes:
        return np.ascontiguousarray(x, dtype=np.float32)

    def decode(self, codes: Codes) -> FloatArray:
        return np.ascontiguousarray(codes, dtype=np.float32)

    def bytes_per_vector(self, dim: int) -> float:
        return 4.0 * dim
