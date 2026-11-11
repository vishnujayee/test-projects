"""The claims analysis must not call noise a win, and must not hide a real win."""

from pathlib import Path

import numpy as np

from trivec.claims import (
    CONTRADICTED,
    INCONCLUSIVE,
    NOT_RUN,
    SUPPORTED,
    evaluate_claims,
    judge,
    load_claims,
    overall,
    paired_bootstrap,
    render_claims_md,
)


def _rec(dataset: str, quantizer: str, recall: list[float], **params: float) -> dict[str, object]:
    return {
        "status": "ok",
        "quantizer": quantizer,
        "quantizer_params": params,
        "search_mode": "asymmetric_flat",
        "dataset": {"name": dataset},
        "per_query": {"recall": recall, "ndcg": recall},
    }


def test_bootstrap_finds_a_real_difference() -> None:
    rng = np.random.default_rng(0)
    b = rng.uniform(0, 0.5, 400)
    mean, low, high = paired_bootstrap(b + 0.1, b)
    assert mean == 0.1 or abs(mean - 0.1) < 1e-9
    assert low > 0 and high > low


def test_bootstrap_does_not_call_noise_a_win() -> None:
    """Many A/A comparisons: the 95% interval should exclude zero only about 5% of the time."""
    false_wins = 0
    for trial in range(200):
        rng = np.random.default_rng(trial)
        a, b = rng.uniform(0, 1, 300), rng.uniform(0, 1, 300)
        _, low, high = paired_bootstrap(a, b, n_boot=300, seed=trial)
        false_wins += low > 0 or high < 0
    assert false_wins <= 25, f"{false_wins}/200 false positives, expected about 10"


def test_bootstrap_ignores_queries_without_labels() -> None:
    a = np.array([1.0, np.nan, 1.0])
    b = np.array([0.0, 0.5, 0.0])
    mean, _, _ = paired_bootstrap(a, b, n_boot=50)
    assert mean == 1.0


def test_judge_with_and_without_margin() -> None:
    assert judge(0.01, 0.03, 0.0) == SUPPORTED
    assert judge(-0.03, -0.01, 0.0) == CONTRADICTED
    assert judge(-0.01, 0.01, 0.0) == INCONCLUSIVE
    assert judge(-0.01, 0.0, 0.02) == SUPPORTED  # within the allowed margin
    assert judge(-0.05, -0.03, 0.02) == CONTRADICTED
    assert judge(-0.03, -0.01, 0.02) == INCONCLUSIVE


def test_overall_requires_every_dataset() -> None:
    assert overall([SUPPORTED, SUPPORTED]) == "supported on every dataset"
    assert overall([SUPPORTED, INCONCLUSIVE]).startswith("mixed")
    assert overall([SUPPORTED, CONTRADICTED]).startswith("mixed")
    assert overall([NOT_RUN]) == NOT_RUN


def test_evaluate_claims_end_to_end() -> None:
    n = 100
    rng = np.random.default_rng(1)
    low = rng.uniform(0, 0.5, n).tolist()
    high = [v + 0.2 for v in low]
    records = [
        _rec("d1", "binary", low),
        _rec("d1", "ternary", high, alpha=0.7),
        _rec("d2", "binary", high),
        _rec("d2", "ternary", low, alpha=0.7),
    ]
    claim = {
        "id": "X",
        "text": "ternary wins",
        "a": {"method": "ternary(alpha=0.7)", "mode": "asymmetric_flat"},
        "b": {"method": "binary", "mode": "asymmetric_flat"},
        "metric": "recall",
    }
    missing = {**claim, "id": "Y", "b": {"method": "pq(m=96)", "mode": "asymmetric_flat"}}
    results = evaluate_claims(records, [claim, missing], n_boot=200)
    verdicts = {r["dataset"]: r["verdict"] for r in results[0]["rows"]}
    assert verdicts == {"d1": SUPPORTED, "d2": CONTRADICTED}
    assert results[0]["overall"].startswith("mixed")
    assert results[1]["overall"] == NOT_RUN
    md = render_claims_md(results)
    assert "### X: ternary wins" in md and "not run" in md and "not a general proof" in md


def test_lab_results_never_count_towards_claims() -> None:
    rec = _rec("d1", "binary", [0.5, 0.5])
    rec["source"] = "lab"
    assert evaluate_claims([rec], [], n_boot=10) == []
    claim = {
        "id": "X",
        "text": "t",
        "a": {"method": "binary", "mode": "asymmetric_flat"},
        "b": {"method": "binary", "mode": "asymmetric_flat"},
        "metric": "recall",
    }
    assert evaluate_claims([rec], [claim], n_boot=10)[0]["rows"] == []


def test_real_claims_file_is_valid() -> None:
    claims = load_claims(Path("configs/claims.yaml"))
    assert len(claims) >= 7 and len({c["id"] for c in claims}) == len(claims)
    for c in claims:
        assert c["metric"] in ("recall", "ndcg")
        assert {"method", "mode"} <= set(c["a"]) and {"method", "mode"} <= set(c["b"])
    assert load_claims(Path("does/not/exist.yaml")) == []
