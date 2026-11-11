"""Exact float32 nearest-neighbour search.

This function IS the ground truth (hard rule 4): every recall number in the project
is measured against what it returns on the same embeddings.
"""

import numpy as np
import numpy.typing as npt

from trivec.quantizers.base import FloatArray, IntArray


def topk_from_scores(scores: npt.NDArray[np.floating], k: int) -> tuple[IntArray, FloatArray]:
    """Best-first top-k ids and scores for each row of a [n_queries, n_corpus] score matrix."""
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    k = min(k, scores.shape[1])
    # argpartition finds the k best without sorting everything (fast); then sort just those k.
    part_idx = np.argpartition(-scores, k - 1, axis=1)[:, :k]
    part_scores = np.take_along_axis(scores, part_idx, axis=1)
    order = np.argsort(-part_scores, axis=1, kind="stable")
    ids = np.take_along_axis(part_idx, order, axis=1).astype(np.int64)
    top = np.take_along_axis(part_scores, order, axis=1).astype(np.float32)
    return ids, top


def exact_search(queries: FloatArray, corpus: FloatArray, k: int) -> tuple[IntArray, FloatArray]:
    """Return the top-k corpus ids and scores for each query by dot product.

    Vectors are expected to be L2-normalised, so dot product equals cosine similarity.
    Results are sorted best-first. Shapes: ids [n_queries, k], scores [n_queries, k].
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    return topk_from_scores(queries @ corpus.T, k)
