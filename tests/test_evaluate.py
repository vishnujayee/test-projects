"""evaluate() on the toy dataset: schema, provenance fields, failure recording."""

import datetime as dt
import json
from pathlib import Path

import numpy as np

from trivec.data import make_toy
from trivec.data.prepare import PreparedData
from trivec.eval import RunSpec, evaluate, make_run_id, write_result
from trivec.eval.provenance import CODE_PATHS
from trivec.quantizers import Float32Quantizer, TernaryQuantizer

REQUIRED_METRICS = {
    "recall_at_10",
    "ndcg_at_10",
    "bytes_per_vector",
    "bytes_per_vector_measured",
    "bytes_per_vector_amortized",
    "shared_state_bytes",
    "p50_latency_ms",
    "p95_latency_ms",
    "index_build_s",
}


def toy_prepared(n: int = 300, dim: int = 64) -> PreparedData:
    toy = make_toy(n_corpus=n, n_queries=30, dim=dim, k=10)
    corpus_ids = [f"d{i}" for i in range(n)]
    query_ids = [f"q{i}" for i in range(30)]
    # the planted source document is the one relevant document for each query
    qrels = {f"q{i}": {f"d{int(s)}": 1} for i, s in enumerate(toy.source_ids)}
    return PreparedData(
        dataset_name="toy",
        dataset_revision="toy-v1",
        model_name="none",
        model_revision="none",
        corpus=toy.corpus,
        queries=toy.queries,
        corpus_ids=corpus_ids,
        query_ids=query_ids,
        qrels=qrels,
        ground_truth=toy.ground_truth,
    )


def test_evaluate_float32_is_perfect_and_complete() -> None:
    data = toy_prepared()
    spec = RunSpec("float32", {}, seed=0, warmup=2, config={"name": "unit"})
    rec = evaluate(spec, data, Float32Quantizer())
    assert rec["status"] == "ok", rec["error"]
    assert set(rec["metrics"]) >= REQUIRED_METRICS
    assert rec["metrics"]["recall_at_10"] == 1.0
    assert rec["metrics"]["ndcg_at_10"] == 1.0  # planted doc is always top-1 on this toy
    assert (
        rec["metrics"]["bytes_per_vector"] == rec["metrics"]["bytes_per_vector_measured"] == 256.0
    )
    for key in ("git_commit", "packages", "hardware", "python", "timestamp"):
        assert key in rec["provenance"]
    assert rec["dataset"]["n_corpus"] == 300 and rec["model"]["dim"] == 64
    assert rec["verification"] == "unverified"
    assert rec["config"] == {"name": "unit"}


def test_evaluate_packed_mode_for_ternary() -> None:
    data = toy_prepared()
    spec = RunSpec("ternary", {"alpha": 0.7}, seed=0, search_mode="symmetric_packed", warmup=2)
    rec = evaluate(spec, data, TernaryQuantizer(alpha=0.7))
    assert rec["status"] == "ok", rec["error"]
    assert 0.0 < rec["metrics"]["recall_at_10"] <= 1.0
    assert rec["run_id"].endswith("_toy_ternary-alpha0.7_packed_s0")


def test_evaluate_records_failures_instead_of_raising() -> None:
    data = toy_prepared()
    spec = RunSpec("float32", {}, seed=0, search_mode="symmetric_packed", warmup=2)
    rec = evaluate(spec, data, Float32Quantizer())  # float32 has no packed search
    assert rec["status"] == "failed"
    assert rec["error"]["type"] == "ValueError"


def test_evaluate_rejects_mismatched_quantizer_name() -> None:
    rec = evaluate(RunSpec("ternary", {}, seed=0, warmup=1), toy_prepared(), Float32Quantizer())
    assert rec["status"] == "failed" and "does not match" in rec["error"]["message"]


def test_run_ids_differ_by_params_and_results_are_never_overwritten(tmp_path: Path) -> None:
    data = toy_prepared(n=100)
    when = dt.datetime(2026, 1, 1, tzinfo=dt.UTC)
    assert make_run_id(RunSpec("t", {"alpha": 0.3}, seed=0), "d", when) != make_run_id(
        RunSpec("t", {"alpha": 0.5}, seed=0), "d", when
    )
    a = evaluate(RunSpec("float32", {}, seed=0, warmup=1), data, Float32Quantizer())
    b = dict(a)  # same id on purpose: two runs finishing in the same second
    first, second = write_result(a, tmp_path), write_result(b, tmp_path)
    assert first != second and first.exists() and second.exists()
    assert json.loads(second.read_text(encoding="utf-8"))["run_id"] == second.stem


def test_per_query_scores_are_recorded_and_match_the_means() -> None:
    data = toy_prepared()
    rec = evaluate(RunSpec("ternary", {"alpha": 0.7}, seed=0, warmup=1), data, TernaryQuantizer())
    pq = rec["per_query"]
    assert len(pq["recall"]) == len(pq["ndcg"]) == 30
    assert abs(float(np.mean(pq["recall"])) - rec["metrics"]["recall_at_10"]) < 1e-5


def test_git_dirty_looks_only_at_code() -> None:
    assert all(p.startswith(":/") for p in CODE_PATHS)
    assert ":/results" not in CODE_PATHS


def test_write_result_roundtrip(tmp_path: Path) -> None:
    data = toy_prepared(n=100)
    rec = evaluate(RunSpec("float32", {}, seed=1, warmup=1), data, Float32Quantizer())
    path = write_result(rec, tmp_path / "results")
    assert path.name == f"{rec['run_id']}.json"
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["seed"] == 1 and loaded["metrics"]["recall_at_10"] == 1.0
    assert not list((tmp_path / "results").glob("*.tmp"))
    assert np.isfinite(loaded["metrics"]["p50_latency_ms"])
