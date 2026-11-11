"""Latency measurement. Part of the evaluator, frozen at the end of Phase 1.

One query at a time (batch size 1), after warm-up queries, because that is what a real
service feels. p50 = median, p95 = slow tail. Also a batched throughput number.
"""

import time
from collections.abc import Callable
from typing import Any

import numpy as np

from trivec.quantizers.base import FloatArray

SearchFn = Callable[[FloatArray, int], Any]


def measure_latency(
    search: SearchFn, queries: FloatArray, k: int, *, warmup: int = 20
) -> dict[str, float]:
    n = int(queries.shape[0])
    for i in range(min(warmup, n)):
        search(queries[i : i + 1], k)

    times_ms = np.empty(n, dtype=np.float64)
    for i in range(n):
        t0 = time.perf_counter()
        search(queries[i : i + 1], k)
        times_ms[i] = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    search(queries, k)
    batched_s = time.perf_counter() - t0

    return {
        "p50_latency_ms": float(np.percentile(times_ms, 50)),
        "p95_latency_ms": float(np.percentile(times_ms, 95)),
        "mean_latency_ms": float(times_ms.mean()),
        "batched_qps": float(n / batched_s) if batched_s > 0 else float("inf"),
    }
