"""`trivec report`: tables, charts and a dashboard generated ONLY from results/*.json.

Hard rule 3: nobody types numbers. This module reads result files and writes
docs/results.md, docs/charts/*.png and docs/dashboard/index.html.
"""

import json
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ResultRecord = dict[str, Any]


def load_results(results_dir: Path) -> list[ResultRecord]:
    records: list[ResultRecord] = []
    for path in sorted(results_dir.glob("*.json")):
        if path.name == "verification.json":
            continue
        records.append(json.loads(path.read_text(encoding="utf-8")))
    return records


def load_verification(results_dir: Path) -> dict[str, str]:
    path = results_dir / "verification.json"
    if not path.exists():
        return {}
    data: dict[str, str] = json.loads(path.read_text(encoding="utf-8"))
    return data


def method_label(rec: ResultRecord) -> str:
    params = rec.get("quantizer_params") or {}
    inner = ", ".join(f"{k}={v}" for k, v in sorted(params.items()))
    return f"{rec['quantizer']}({inner})" if inner else str(rec["quantizer"])


@dataclass
class Row:
    method: str
    mode: str
    n_seeds: int
    metrics: dict[str, tuple[float, float]]  # name -> (mean, std)
    run_ids: list[str]


GroupKey = tuple[str, str, int]  # dataset, model, k
BASELINES = {"float32", "int8", "binary", "pq"}
DEV_DATASET = "scifact"
PRIMARY_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def aggregate(records: list[ResultRecord]) -> dict[GroupKey, list[Row]]:
    """Group ok runs by (dataset, model, k), then by (method, mode); mean ± std over seeds."""
    buckets: dict[GroupKey, dict[tuple[str, str], list[ResultRecord]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for rec in records:
        if rec.get("status") != "ok":
            continue
        key: GroupKey = (rec["dataset"]["name"], rec["model"]["name"], int(rec["k"]))
        buckets[key][(method_label(rec), rec["search_mode"])].append(rec)

    out: dict[GroupKey, list[Row]] = {}
    for key, groups in buckets.items():
        rows: list[Row] = []
        for (method, mode), recs in groups.items():
            names = sorted({m for r in recs for m in r["metrics"]})
            metrics: dict[str, tuple[float, float]] = {}
            for name in names:
                vals = [float(r["metrics"][name]) for r in recs if name in r["metrics"]]
                mean = statistics.fmean(vals)
                std = statistics.pstdev(vals) if len(vals) > 1 else 0.0
                metrics[name] = (mean, std)
            rows.append(Row(method, mode, len(recs), metrics, [r["run_id"] for r in recs]))
        rows.sort(key=lambda r: (-r.metrics.get("bytes_per_vector", (0, 0))[0], r.method, r.mode))
        out[key] = rows
    return out


def _fmt(row: Row, name: str, digits: int = 4) -> str:
    if name not in row.metrics:
        return "—"
    mean, std = row.metrics[name]
    if std > 0:
        return f"{mean:.{digits}f} ± {std:.{digits}f}"
    return f"{mean:.{digits}f}"


def markdown_table(key: GroupKey, rows: list[Row]) -> str:
    _dataset, _model, k = key
    head = (
        f"| method | mode | seeds | recall@{k} | nDCG@{k} | bytes/vector | with shared state | p50 ms | p95 ms | build s |\n"
        "|---|---|---|---|---|---|---|---|---|---|\n"
    )
    body = "".join(
        f"| {r.method} | {r.mode} | {r.n_seeds} | {_fmt(r, f'recall_at_{k}')} | {_fmt(r, f'ndcg_at_{k}')} | "
        f"{_fmt(r, 'bytes_per_vector', 0)} | {_fmt(r, 'bytes_per_vector_amortized', 1)} | {_fmt(r, 'p50_latency_ms', 2)} | {_fmt(r, 'p95_latency_ms', 2)} | "
        f"{_fmt(r, 'index_build_s', 2)} |\n"
        for r in rows
    )
    return head + body


SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e7e6e1"
BASELINE_GRAY = "#8a8983"
TERNARY_BLUE = "#2a78d6"  # categorical slot 1
LAB_ORANGE = "#eb6834"  # categorical slot 2
HEADLINE_ALPHA = "ternary(alpha=0.7)"
LABEL_SLOTS = [
    (7, 3, "left"),
    (7, -9, "left"),
    (-7, 3, "right"),
    (-7, -9, "right"),
    (7, 14, "left"),
    (7, -20, "left"),
    (-7, 14, "right"),
    (-7, -20, "right"),
    (7, 25, "left"),
    (7, -31, "left"),
    (-7, 25, "right"),
    (-7, -31, "right"),
]


def chart_suffix(key: GroupKey, groups: dict[GroupKey, list[Row]]) -> str:
    """The dataset name alone, unless the dataset was also run with another embedding model."""
    dataset, model, _k = key
    models = {m for d, m, _ in groups if d == dataset}
    return (
        dataset
        if len(models) == 1 or model == PRIMARY_MODEL
        else f"{dataset}_{model.split('/')[-1]}"
    )


def _chart_rows(rows: list[Row]) -> list[Row]:
    """The alpha sweep has its own table; on a chart it would hide the comparison that matters."""
    return [r for r in rows if not r.method.startswith("ternary(") or r.method == HEADLINE_ALPHA]


def _series(r: Row) -> str:
    if r.method.startswith("ternary"):
        return "ternary"
    return "baseline" if r.method.split("(")[0] in BASELINES else "lab"


def _scatter(rows: list[Row], x_metric: str, y_metric: str, ax: Any) -> None:
    """Dots with a surface ring. Every dot is labelled in ink, so color never carries identity alone."""
    colors = {"baseline": BASELINE_GRAY, "ternary": TERNARY_BLUE, "lab": LAB_ORANGE}
    names = {"baseline": "Baselines", "ternary": "Ternary", "lab": "Lab experiments"}
    placed: list[tuple[float, float, float, float]] = []
    pts = [r for r in rows if x_metric in r.metrics and y_metric in r.metrics]
    pts.sort(key=lambda r: -r.metrics[y_metric][0])
    seen: set[str] = set()
    scale = ax.figure.dpi / 72
    for r in pts:
        x, y = r.metrics[x_metric][0], r.metrics[y_metric][0]
        series = _series(r)
        packed = r.mode == "symmetric_packed"
        ax.scatter(
            x,
            y,
            s=64,
            marker="s" if packed else "o",
            color=colors[series],
            edgecolors=SURFACE,
            linewidths=2,
            zorder=3,
            label=names[series] if series not in seen else None,
        )
        seen.add(series)
        text = f"{r.method} packed" if packed else r.method
        px, py = ax.transData.transform((x, y))
        width, height = len(text) * 4.6 * scale, 9 * scale
        for dx, dy, ha in LABEL_SLOTS:
            x0 = px + dx * scale if ha == "left" else px + dx * scale - width
            box = (x0, py + dy * scale, x0 + width, py + dy * scale + height)
            clash = any(
                box[0] < p[2] and p[0] < box[2] and box[1] < p[3] and p[1] < box[3] for p in placed
            )
            if not clash:
                break
        placed.append(box)
        ax.annotate(
            text,
            (x, y),
            xytext=(dx, dy),
            textcoords="offset points",
            ha=ha,
            fontsize=8,
            color=INK_SECONDARY,
        )


def _style(ax: Any, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.figure.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    ax.set_xlabel(xlabel, fontsize=9, color=INK_SECONDARY)
    ax.set_ylabel(ylabel, fontsize=9, color=INK_SECONDARY)
    ax.tick_params(colors=INK_SECONDARY, labelsize=8, length=0)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    legend = ax.legend(
        frameon=False,
        fontsize=8,
        loc="lower right",
        title="Circle: flat. Square: packed.",
        title_fontsize=7,
    )
    for text in legend.get_texts():
        text.set_color(INK_SECONDARY)
    legend.get_title().set_color(INK_SECONDARY)


def write_charts(groups: dict[GroupKey, list[Row]], charts_dir: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    charts_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    specs = [
        (
            "recall_vs_bytes",
            "bytes_per_vector",
            "bytes per vector, codes only (log scale)",
            "recall vs memory",
        ),
        (
            "latency_vs_recall",
            "p50_latency_ms",
            "median ms per query (log scale). Flat decodes the corpus, so it is not a speed claim",
            "recall vs time",
        ),
    ]
    for key, all_rows in groups.items():
        k = key[2]
        rows = _chart_rows(all_rows)
        suffix = chart_suffix(key, groups)
        for stem, x_metric, xlabel, title in specs:
            fig, ax = plt.subplots(figsize=(7.2, 4.6))
            ax.set_xscale("log")
            xs = [r.metrics[x_metric][0] for r in rows if x_metric in r.metrics]
            if xs and min(xs) > 0:
                ax.set_xlim(min(xs) / 1.6, max(xs) * 2.2)
            ax.set_ylim(0, 1.05)
            fig.tight_layout()
            _scatter(rows, x_metric, f"recall_at_{k}", ax)
            _style(ax, f"{suffix}: {title}", xlabel, f"recall@{k} against exact float32")
            path = charts_dir / f"{stem}_{suffix}.png"
            fig.tight_layout()
            fig.savefig(path, dpi=130, facecolor=SURFACE)
            plt.close(fig)
            written.append(path)
    return written


def above_float32_section(groups: dict[GroupKey, list[Row]], records: list[ResultRecord]) -> str:
    """A compressed method scoring above uncompressed float32 looks too good. Test each one."""
    from trivec.claims import paired_bootstrap, per_query_scores

    rows_md: list[str] = []
    for (dataset, model, k), rows in sorted(groups.items()):
        float32 = next((r for r in rows if r.method == "float32"), None)
        base = per_query_scores(records, (dataset, model), "float32", "asymmetric_flat", "ndcg")
        if float32 is None or base is None:
            continue
        for r in rows:
            if (
                r.method == "float32"
                or r.metrics[f"ndcg_at_{k}"][0] <= float32.metrics[f"ndcg_at_{k}"][0]
            ):
                continue
            a = per_query_scores(records, (dataset, model), r.method, r.mode, "ndcg")
            if a is None:
                continue
            mean, low, high = paired_bootstrap(a, base)
            verdict = "real" if low > 0 else "within noise"
            rows_md.append(
                f"| {dataset} ({model.split('/')[-1]}) | {r.method} {r.mode} | {mean:+.4f} | [{low:+.4f}, {high:+.4f}] | {verdict} |\n"
            )
    if not rows_md:
        return ""
    return (
        "## Compressed methods that scored above float32\n\n"
        "Compression throws information away, so a higher nDCG than float32 needs explaining. "
        "Each row is nDCG minus float32, with a 95% paired bootstrap interval over queries.\n\n"
        "| dataset | method | difference | 95% interval | reading |\n|---|---|---|---|---|\n"
        + "".join(rows_md)
        + "\n"
    )


def alpha_section(groups: dict[GroupKey, list[Row]]) -> str:
    """Alpha is chosen on the development dataset only, then read on the others."""
    table: dict[str, dict[str, float]] = {}
    primary_only = any(m == PRIMARY_MODEL for _, m, _ in groups)
    for (dataset, model, k), rows in groups.items():
        if primary_only and model != PRIMARY_MODEL:
            continue
        for r in rows:
            if r.method.startswith("ternary(alpha=") and r.mode == "asymmetric_flat":
                table.setdefault(r.method, {})[dataset] = r.metrics[f"recall_at_{k}"][0]
    datasets = sorted({d for by_ds in table.values() for d in by_ds})
    if len(table) < 2 or DEV_DATASET not in datasets:
        return ""
    best = max((m for m in table if DEV_DATASET in table[m]), key=lambda m: table[m][DEV_DATASET])
    lines = [
        "## Choosing alpha\n\n",
        f"Recall@10 of ternary (asymmetric_flat) by alpha. Alpha is selected on `{DEV_DATASET}` only, "
        "so the other datasets stay a fair test of that choice. "
        f"Selected: **{best}**.\n\n",
        "| alpha | " + " | ".join(datasets) + " |\n",
        "|---|" + "---|" * len(datasets) + "\n",
    ]
    for method in sorted(table):
        cells = " | ".join(
            f"{table[method][d]:.4f}" if d in table[method] else "—" for d in datasets
        )
        lines.append(f"| {method} | {cells} |\n")
    lines.append("\n")
    return "".join(lines)


def write_results_md(
    groups: dict[GroupKey, list[Row]],
    records: list[ResultRecord],
    out_path: Path,
    charts_rel: str = "charts",
    claims_md: str = "",
    break_md: str = "",
) -> None:
    n_ok = sum(1 for r in records if r.get("status") == "ok")
    n_failed = len(records) - n_ok
    parts: list[str] = [
        "# Results\n\n",
        "**Generated by `uv run trivec report` from `results/*.json`. Do not edit numbers by hand (hard rule 3).**\n\n",
        f"Runs: {n_ok} ok, {n_failed} failed. Every number is a mean over seeds, with ± population std when seeds differ. "
        "Only PQ and timings vary by seed: binary, int8 and ternary use no randomness, so their quality has no ±. "
        "See Claims tested for real uncertainty over queries.\n\n",
        "recall@k is agreement with exact float32 search on the same embeddings; nDCG@k uses the dataset's human relevance labels.\n",
        "`asymmetric_flat` keeps the query in float32 and decodes the corpus (quality reference). "
        "`symmetric_packed` compresses both sides and searches with popcount (speed path).\n\n",
        "**bytes/vector** counts the stored codes. **with shared state** adds codebooks and scales divided by the corpus size, "
        "which is large for PQ on a small corpus and shrinks as the corpus grows.\n\n",
        "Timings come from one busy laptop running other programs: compare them within a table, do not quote them as absolute.\n\n",
        claims_md,
        break_md,
        above_float32_section(groups, records),
        alpha_section(groups),
    ]
    if not groups:
        parts.append("No successful runs yet.\n")
    for key, rows in sorted(groups.items()):
        dataset, model, k = key
        sample = next(
            r
            for r in records
            if r.get("status") == "ok"
            and r["dataset"]["name"] == dataset
            and r["model"]["name"] == model
        )
        cfg_names = sorted(
            {
                str((r.get("config") or {}).get("name"))
                for r in records
                if r.get("status") == "ok"
                and r.get("source") != "lab"
                and r["dataset"]["name"] == dataset
                and r["model"]["name"] == model
            }
        )
        commands = " && ".join(f"uv run trivec bench --config configs/{c}.yaml" for c in cfg_names)
        hw = sample["provenance"]["hardware"]
        parts.append(f"## {dataset} · {model} · k = {k}\n\n")
        parts.append(
            f"Corpus: {sample['dataset']['n_corpus']} docs, {sample['dataset']['n_queries']} queries. "
            f"Dataset revision `{str(sample['dataset']['revision'])[:12]}`, model revision "
            f"`{str(sample['model']['revision'])[:12]}`. Machine: {hw.get('cpu')} "
            f"({hw.get('cpu_count')} threads, OMP_NUM_THREADS={hw.get('omp_num_threads')}).\n\n"
        )
        parts.append(markdown_table(key, rows))
        suffix = chart_suffix(key, groups)
        parts.append(f"\n![recall vs bytes]({charts_rel}/recall_vs_bytes_{suffix}.png)\n")
        parts.append(f"![latency vs recall]({charts_rel}/latency_vs_recall_{suffix}.png)\n\n")
        parts.append(f"Reproduce: `{commands} && uv run trivec report`\n\n")
        run_ids = sorted({rid for r in rows for rid in r.run_ids})
        parts.append(
            "<details><summary>run ids</summary>\n\n"
            + "\n".join(f"- `{rid}`" for rid in run_ids)
            + "\n\n</details>\n\n"
        )
    failed = [r for r in records if r.get("status") != "ok"]
    if failed:
        parts.append("## Failed runs (kept on purpose, hard rule 9)\n\n")
        for r in failed:
            err = r.get("error") or {}
            parts.append(f"- `{r.get('run_id')}`: {err.get('type')}: {err.get('message')}\n")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(parts), encoding="utf-8")


def write_dashboard(
    records: list[ResultRecord], verification: dict[str, str], out_path: Path
) -> None:
    """A static leaderboard: every run, labelled unverified / reproduced / verified."""
    rows = []
    for r in sorted(records, key=lambda x: x["run_id"], reverse=True):
        m = r.get("metrics") or {}
        k = r.get("k", 10)
        label = verification.get(r["run_id"], r.get("verification", "unverified"))
        rows.append(
            "<tr>"
            f"<td>{r['run_id']}</td><td>{(r.get('dataset') or {}).get('name', '?')}</td><td>{method_label(r)}</td><td>{r.get('search_mode', '?')}</td>"
            f"<td>{r.get('status')}</td><td>{m.get(f'recall_at_{k}', float('nan')):.4f}</td>"
            f"<td>{m.get(f'ndcg_at_{k}', float('nan')):.4f}</td><td>{m.get('bytes_per_vector', float('nan')):.0f}</td>"
            f"<td>{m.get('p50_latency_ms', float('nan')):.2f}</td><td class='{label}'>{label}</td>"
            "</tr>"
        )
    html = (
        "<!doctype html><html><head><meta charset='utf-8'><title>TriVec Lab leaderboard</title>"
        "<style>body{font-family:system-ui;margin:2rem;background:#fff;color:#111}"
        "table{border-collapse:collapse;font-size:13px}td,th{border:1px solid #ddd;padding:4px 8px}"
        "th{background:#f3f3f3}.unverified{color:#a60}.reproduced{color:#06a}.verified{color:#080;font-weight:bold}"
        "</style></head><body><h1>TriVec Lab leaderboard</h1>"
        "<p>Generated from <code>results/</code>. Labels: <span class='unverified'>unverified</span> (one run), "
        "<span class='reproduced'>reproduced</span> (new seeds + held-out dataset), "
        "<span class='verified'>verified</span> (manual approval). The lab never announces discoveries itself.</p>"
        "<table><tr><th>run</th><th>dataset</th><th>method</th><th>mode</th><th>status</th><th>recall@k</th>"
        "<th>nDCG@k</th><th>bytes/vec</th><th>p50 ms</th><th>label</th></tr>"
        + "".join(rows)
        + "</table></body></html>"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")


def run_report(
    results_dir: Path,
    docs_dir: Path,
    *,
    dashboard: bool = True,
    claims_path: Path = Path("configs/claims.yaml"),
) -> dict[str, Any]:
    from trivec.claims import evaluate_claims, load_claims, render_claims_md

    records = load_results(results_dir)
    groups = aggregate(records)
    charts = write_charts(groups, docs_dir / "charts")
    from trivec.break_tests import render_break_md

    claims_md = render_claims_md(evaluate_claims(records, load_claims(claims_path)))
    break_path = results_dir / "checks" / "break_tests.json"
    break_md = (
        render_break_md(json.loads(break_path.read_text(encoding="utf-8")))
        if break_path.exists()
        else ""
    )
    write_results_md(
        groups, records, docs_dir / "results.md", claims_md=claims_md, break_md=break_md
    )
    if dashboard:
        write_dashboard(
            records, load_verification(results_dir), docs_dir / "dashboard" / "index.html"
        )
    return {"n_records": len(records), "n_groups": len(groups), "charts": [str(c) for c in charts]}
