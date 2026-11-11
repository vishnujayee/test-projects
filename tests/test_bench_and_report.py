"""bench + report on the toy dataset, end to end, in a temp directory."""

import json
from pathlib import Path

import yaml

from tests.test_evaluate import toy_prepared
from trivec.bench import iter_specs, load_config, run_bench
from trivec.report import aggregate, load_results, method_label, run_report

CONFIG = {
    "name": "toy_unit",
    "dataset": "toy",
    "model": {"name": "none", "revision": None},
    "k": 10,
    "seeds": [0, 1],
    "warmup": 1,
    "quantizers": [
        {"name": "float32"},
        {"name": "binary", "modes": ["asymmetric_flat", "symmetric_packed"]},
        {"name": "ternary", "params": {"alpha": 0.7}, "modes": ["symmetric_packed"]},
        {"name": "float32", "modes": ["symmetric_packed"]},  # will fail on purpose
    ],
}


def _write_config(tmp_path: Path) -> Path:
    path = tmp_path / "toy.yaml"
    path.write_text(yaml.safe_dump(CONFIG), encoding="utf-8")
    return path


def test_iter_specs_expands_modes_and_seeds(tmp_path: Path) -> None:
    cfg = load_config(_write_config(tmp_path))
    specs = list(iter_specs(cfg))
    # float32: 1 mode x 2 seeds, binary: 2 x 2, ternary: 1 x 2, float32 packed: 1 x 2
    assert len(specs) == 2 + 4 + 2 + 2
    assert specs[0].config["name"] == "toy_unit"


def test_bench_then_report(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    written = run_bench(_write_config(tmp_path), results_dir=results_dir, data=toy_prepared(n=200))
    assert len(written) == 10 and all(p.exists() for p in written)

    records = load_results(results_dir)
    statuses = [r["status"] for r in records]
    assert statuses.count("ok") == 8 and statuses.count("failed") == 2

    groups = aggregate(records)
    assert list(groups) == [("toy", "none", 10)]
    rows = groups[("toy", "none", 10)]
    labels = {(r.method, r.mode) for r in rows}
    assert labels == {
        ("float32", "asymmetric_flat"),
        ("binary", "asymmetric_flat"),
        ("binary", "symmetric_packed"),
        ("ternary(alpha=0.7)", "symmetric_packed"),
    }
    assert all(r.n_seeds == 2 for r in rows)
    assert rows[0].method == "float32"  # sorted by bytes/vector, biggest first

    docs = tmp_path / "docs"
    info = run_report(results_dir, docs)
    assert info["n_records"] == 10 and info["n_groups"] == 1
    md = (docs / "results.md").read_text(encoding="utf-8")
    assert "## toy · none · k = 10" in md
    assert "ternary(alpha=0.7)" in md and "Failed runs" in md
    assert "Reproduce: `uv run trivec bench --config configs/toy_unit.yaml" in md
    assert (docs / "charts" / "recall_vs_bytes_toy.png").exists()
    assert (docs / "charts" / "latency_vs_recall_toy.png").exists()
    html = (docs / "dashboard" / "index.html").read_text(encoding="utf-8")
    assert html.count("unverified") >= 10


def test_method_label() -> None:
    assert method_label({"quantizer": "pq", "quantizer_params": {"m": 48}}) == "pq(m=48)"
    assert method_label({"quantizer": "binary", "quantizer_params": {}}) == "binary"
    rec = json.loads(json.dumps({"quantizer": "ternary", "quantizer_params": {"alpha": 0.7}}))
    assert method_label(rec) == "ternary(alpha=0.7)"


def test_break_tests_on_toy(tmp_path: Path) -> None:
    """The checks must behave as designed: a leak-free pipeline collapses on shuffled labels."""
    from trivec.break_tests import (
        recall_without_self,
        render_break_md,
        rotation,
        scale_variants,
        shuffled_labels,
        threshold_theory,
        tie_order,
    )

    data = toy_prepared(n=300)
    labels = shuffled_labels(data)
    assert labels["real_ndcg"] == 1.0 and labels["shuffled_ndcg"] < 0.5
    rot = rotation(data, n_rotations=1)
    assert rot["float32"]["rotated_mean"] > 0.999
    selfish = recall_without_self(data)
    assert selfish["queries_also_in_corpus"] == 0
    assert abs(selfish["binary"]["with_self"] - selfish["binary"]["without_self"]) < 1e-9
    theory = threshold_theory(data)
    assert 0.70 < theory["lloyd_alpha_gaussian"] < 0.80  # 3-level Gaussian: 0.612 / 0.798 = 0.767
    report = {
        "datasets": {
            "toy": {
                "shuffled_labels": labels,
                "recall_without_self": selfish,
                "rotation": rot,
                "tie_order": tie_order(data),
                "scale_variants": scale_variants(data),
                "threshold_theory": theory,
            }
        }
    }
    md = render_break_md(report)
    assert "Shuffled labels" in md and "| toy |" in md
    assert render_break_md({"datasets": {}}) == ""


def test_two_models_on_one_dataset_keep_separate_charts(tmp_path: Path) -> None:
    from trivec.report import PRIMARY_MODEL, chart_suffix

    groups = {("d", PRIMARY_MODEL, 10): [], ("d", "org/other", 10): [], ("e", "org/other", 10): []}
    assert chart_suffix(("d", PRIMARY_MODEL, 10), groups) == "d"
    assert chart_suffix(("d", "org/other", 10), groups) == "d_other"
    assert chart_suffix(("e", "org/other", 10), groups) == "e"
