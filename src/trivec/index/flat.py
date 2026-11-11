"""FlatIndex: the universal, slow, obviously-correct search.

Asymmetric: the query stays float32; the corpus is decoded from its codes once at
build time. Works for ANY quantizer because every quantizer has `decode`.

Honesty note: this decodes the whole corpus to float32 in RAM, so its latency is a
reference number, not a speed claim. Memory claims refer to the stored codes.
"""

from trivec.index.exact import topk_from_scores
from trivec.quantizers.base import Codes, FloatArray, IntArray, Quantizer


class FlatIndex:
    mode = "asymmetric_flat"

    def __init__(self, quantizer: Quantizer, corpus_codes: Codes) -> None:
        self.quantizer = quantizer
        self.codes = corpus_codes
        self._decoded: FloatArray = quantizer.decode(corpus_codes)

    def __len__(self) -> int:
        return int(self._decoded.shape[0])

    def search(self, queries: FloatArray, k: int) -> tuple[IntArray, FloatArray]:
        return topk_from_scores(queries @ self._decoded.T, k)
