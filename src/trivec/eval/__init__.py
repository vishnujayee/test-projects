"""The evaluator: metrics, latency, provenance, evaluate(). FROZEN at the end of Phase 1."""

from trivec.eval.evaluate import RunSpec, evaluate, make_run_id, write_result
from trivec.eval.latency import measure_latency
from trivec.eval.metrics import dcg, ndcg_at_k, recall_at_k

__all__ = [
    "RunSpec",
    "dcg",
    "evaluate",
    "make_run_id",
    "measure_latency",
    "ndcg_at_k",
    "recall_at_k",
    "write_result",
]
