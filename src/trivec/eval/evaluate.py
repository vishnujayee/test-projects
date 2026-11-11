"""evaluate(): run one quantizer on one prepared dataset and produce a ResultRecord.

This is the judge. Frozen at the end of Phase 1 (hash-checked, see docs/design.md 6.4).
Agent code never imports this module; the Lab runner calls it from a subprocess.
"""

import datetime as dt
import json
import re
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from trivec.data.prepare import PreparedData
from trivec.eval.latency import measure_latency
from trivec.eval.memory import shared_state_bytes
from trivec.eval.metrics import ndcg_per_query, recall_per_query
from trivec.eval.provenance import collect
from trivec.index.flat import FlatIndex
from trivec.index.packed import PackedPopcountIndex
from trivec.quantizers.base import Quantizer
from trivec.quantizers.binary import BinaryQuantizer
from trivec.quantizers.ternary import TernaryQuantizer

SEARCH_MODES = ("asymmetric_flat", "symmetric_packed")

ResultRecord = dict[str, Any]


@dataclass(frozen=True)
class RunSpec:
    quantizer_name: str
    quantizer_params: dict[str, Any]
    seed: int
    k: int = 10
    search_mode: str = "asymmetric_flat"
    warmup: int = 20
    config: dict[str, Any] = field(default_factory=dict)
    """The full experiment config this run came from, copied into the result."""

    def __post_init__(self) -> None:
        if self.search_mode not in SEARCH_MODES:
            raise ValueError(f"search_mode must be one of {SEARCH_MODES}")


def make_run_id(spec: RunSpec, dataset: str, when: dt.datetime | None = None) -> str:
    when = when or dt.datetime.now(dt.UTC)
    stamp = when.strftime("%Y%m%dT%H%M%S")
    mode = "flat" if spec.search_mode == "asymmetric_flat" else "packed"
    # parameters are part of the id: two alphas finishing in the same second must not collide
    slug = "".join(f"-{k}{v}" for k, v in sorted(spec.quantizer_params.items()))
    slug = re.sub(r"[^A-Za-z0-9.\-]", "", slug)[:40]
    return f"{stamp}_{dataset}_{spec.quantizer_name}{slug}_{mode}_s{spec.seed}"


def _build_index(
    spec: RunSpec, quantizer: Quantizer, codes: Any
) -> FlatIndex | PackedPopcountIndex:
    if spec.search_mode == "symmetric_packed":
        if not isinstance(quantizer, BinaryQuantizer | TernaryQuantizer):
            raise ValueError("symmetric_packed search only supports binary and ternary codes")
        return PackedPopcountIndex(quantizer, codes)
    return FlatIndex(quantizer, codes)


def evaluate(spec: RunSpec, data: PreparedData, quantizer: Quantizer) -> ResultRecord:
    """Fit -> encode -> index -> search -> metrics -> provenance.

    Never raises: a failure is recorded as a result with status "failed" (hard rule 9).
    """
    started = dt.datetime.now(dt.UTC)
    record: ResultRecord = {
        "run_id": make_run_id(spec, data.dataset_name, started),
        "config": spec.config,
        "quantizer": spec.quantizer_name,
        "quantizer_params": spec.quantizer_params,
        "seed": spec.seed,
        "k": spec.k,
        "dataset": {
            "name": data.dataset_name,
            "revision": data.dataset_revision,
            "n_corpus": int(data.corpus.shape[0]),
            "n_queries": int(data.queries.shape[0]),
        },
        "model": {"name": data.model_name, "revision": data.model_revision, "dim": data.dim},
        "search_mode": spec.search_mode,
        "metrics": {},
        "provenance": collect(),
        "status": "ok",
        "error": None,
        "verification": "unverified",
    }
    try:
        if quantizer.name != spec.quantizer_name:
            raise ValueError(
                f"quantizer {quantizer.name!r} does not match spec {spec.quantizer_name!r}"
            )

        t0 = time.perf_counter()
        quantizer.fit(data.corpus, seed=spec.seed)
        codes = quantizer.encode(data.corpus)
        index = _build_index(spec, quantizer, codes)
        build_s = time.perf_counter() - t0

        # k+1 results: nDCG drops a hit whose id equals the query id, and must still have k left
        ids, _ = index.search(data.queries, spec.k + 1)
        n = int(data.corpus.shape[0])
        recalls = recall_per_query(ids, data.ground_truth[:, : spec.k])
        ndcgs = ndcg_per_query(ids, data.query_ids, data.corpus_ids, data.qrels, spec.k)
        counted = [s for s in ndcgs if s is not None]
        shared = shared_state_bytes(quantizer)
        record["metrics"] = {
            f"recall_at_{spec.k}": float(np.mean(recalls)) if recalls else 0.0,
            f"ndcg_at_{spec.k}": float(np.mean(counted)) if counted else 0.0,
            "bytes_per_vector": float(quantizer.bytes_per_vector(data.dim)),
            "bytes_per_vector_measured": float(codes.nbytes / n),
            "shared_state_bytes": float(shared),
            "bytes_per_vector_amortized": float(codes.nbytes / n + shared / n),
            "index_build_s": float(build_s),
            **measure_latency(index.search, data.queries, spec.k, warmup=spec.warmup),
        }
        # one value per query, same order as the dataset: lets the report compare two methods
        # query by query, which is the only uncertainty estimate deterministic methods have
        record["per_query"] = {
            "recall": [round(r, 6) for r in recalls],
            "ndcg": [None if s is None else round(s, 6) for s in ndcgs],
        }
    except Exception as exc:
        record["status"] = "failed"
        record["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
    return record


def write_result(record: ResultRecord, results_dir: Path) -> Path:
    """Atomic write of one result JSON; the file name is the run id. Never overwrites a result."""
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / f"{record['run_id']}.json"
    attempt = 1
    while path.exists():
        attempt += 1
        record["run_id"] = f"{record['run_id'].rsplit('__', 1)[0]}__{attempt}"
        path = results_dir / f"{record['run_id']}.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)
    return path
