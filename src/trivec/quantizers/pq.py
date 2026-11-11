"""Product quantization (PQ) baseline, using the FAISS implementation.

The vector is cut into `m` equal chunks. For each chunk position, k-means learns 2^nbits
"typical chunks" (a codebook). Each vector is then stored as m small codebook ids.
With nbits=8 that is exactly m bytes per vector. Codebooks are shared (O(m * 2^nbits * d/m))
and are not counted in bytes_per_vector.
"""

from typing import Any

import numpy as np

from trivec.quantizers.base import Codes, FloatArray


class PQQuantizer:
    name = "pq"

    def __init__(self, m: int = 48, nbits: int = 8) -> None:
        if nbits != 8:
            raise ValueError("only nbits=8 is supported (one byte per sub-vector)")
        self.m = int(m)
        self.nbits = int(nbits)
        self._pq: Any = None  # faiss.ProductQuantizer, created in fit

    def fit(self, corpus: FloatArray, *, seed: int) -> None:
        import faiss

        d = int(corpus.shape[1])
        if d % self.m != 0:
            raise ValueError(f"dim {d} must be divisible by m={self.m}")
        pq = faiss.ProductQuantizer(d, self.m, self.nbits)
        pq.cp.seed = int(seed)
        pq.verbose = False
        threads = faiss.omp_get_max_threads()
        faiss.omp_set_num_threads(1)  # single-threaded k-means is deterministic
        try:
            pq.train(np.ascontiguousarray(corpus, dtype=np.float32))
        finally:
            faiss.omp_set_num_threads(threads)
        self._pq = pq
        # a numpy copy of the codebooks, so the evaluator can count their memory
        self._centroids = faiss.vector_to_array(pq.centroids)

    def encode(self, x: FloatArray) -> Codes:
        if self._pq is None:
            raise RuntimeError("call fit() before encode()")
        codes: Codes = self._pq.compute_codes(np.ascontiguousarray(x, dtype=np.float32))
        return codes

    def decode(self, codes: Codes) -> FloatArray:
        if self._pq is None:
            raise RuntimeError("call fit() before decode()")
        out: FloatArray = self._pq.decode(np.ascontiguousarray(codes, dtype=np.uint8))
        return np.ascontiguousarray(out, dtype=np.float32)

    def bytes_per_vector(self, dim: int) -> float:
        return float(self.m * self.nbits / 8)
