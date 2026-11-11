"""Test claims declared in configs/claims.yaml against results/*.json, query by query.

Why this exists: binary, int8 and ternary use no randomness, so running three seeds gives the
same recall three times and a std of 0. That says nothing about how sure we should be. The
real uncertainty is "would this hold on other queries?", so we resample the queries.

Paired bootstrap: for each query take (score of A - score of B), resample the queries with
replacement many times, and read a 95% interval for the mean difference.

A claim is only as strong as what was tested. These are the datasets and the one embedding
model in the results, not all data. A "supported" claim is evidence, not proof.
"""

from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import yaml

SUPPORTED = "supported"
CONTRADICTED = "contradicted"
INCONCLUSIVE = "inconclusive"
NOT_RUN = "not run"

FloatVec = npt.NDArray[np.float64]


def load_claims(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return list(data.get("claims", []))


def _label(rec: dict[str, Any]) -> str:
    params = rec.get("quantizer_params") or {}
    inner = ", ".join(f"{k}={v}" for k, v in sorted(params.items()))
    return f"{rec['quantizer']}({inner})" if inner else str(rec["quantizer"])


def _model(rec: dict[str, Any]) -> str:
    return str((rec.get("model") or {}).get("name", "?"))


def per_query_scores(
    records: list[dict[str, Any]],
    target: tuple[str, str],
    method: str,
    mode: str,
    metric: str,
) -> FloatVec | None:
    """Per-query scores for one method, averaged over its seeds. None if it was not run."""
    rows = [
        [np.nan if v is None else float(v) for v in r["per_query"][metric]]
        for r in records
        if r.get("status") == "ok"
        and r.get("source") != "lab"
        and (r["dataset"]["name"], _model(r)) == target
        and r.get("search_mode") == mode
        and _label(r) == method
        and "per_query" in r
    ]
    if not rows:
        return None
    return np.asarray(rows, dtype=np.float64).mean(axis=0)


def paired_bootstrap(
    a: FloatVec, b: FloatVec, *, n_boot: int = 2000, seed: int = 12345
) -> tuple[float, float, float]:
    """Return (mean difference a - b, lower 95%, upper 95%), resampling queries."""
    diff = a - b
    diff = diff[~np.isnan(diff)]
    if diff.size == 0:
        raise ValueError("no comparable queries")
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):  # one resample at a time keeps memory flat on small machines
        means[i] = diff[rng.integers(0, diff.size, diff.size)].mean()
    low, high = np.percentile(means, [2.5, 97.5])
    return float(diff.mean()), float(low), float(high)


def judge(low: float, high: float, margin: float) -> str:
    """ "A is better than B", or with a margin, "A is at most `margin` worse than B"."""
    if low > -margin:
        return SUPPORTED
    if high < -margin:
        return CONTRADICTED
    return INCONCLUSIVE


def overall(verdicts: list[str]) -> str:
    if not verdicts or all(v == NOT_RUN for v in verdicts):
        return NOT_RUN
    if all(v == SUPPORTED for v in verdicts):
        return "supported on every dataset"
    if all(v == CONTRADICTED for v in verdicts):
        return "contradicted on every dataset"
    return "mixed: not supported on every dataset"


def targets_in(records: list[dict[str, Any]]) -> list[tuple[str, str]]:
    """Every (dataset, embedding model) pair. Scores from two models are never compared."""
    return sorted(
        {
            (r["dataset"]["name"], _model(r))
            for r in records
            if r.get("status") == "ok" and r.get("source") != "lab" and "per_query" in r
        }
    )


def evaluate_claims(
    records: list[dict[str, Any]], claims: list[dict[str, Any]], *, n_boot: int = 2000
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    targets = targets_in(records)
    several_models = len({m for _, m in targets}) > 1
    for claim in claims:
        rows: list[dict[str, Any]] = []
        for target in targets:
            ds = f"{target[0]} / {target[1].split('/')[-1]}" if several_models else target[0]
            a = per_query_scores(
                records, target, claim["a"]["method"], claim["a"]["mode"], claim["metric"]
            )
            b = per_query_scores(
                records, target, claim["b"]["method"], claim["b"]["mode"], claim["metric"]
            )
            if a is None or b is None:
                rows.append({"dataset": ds, "verdict": NOT_RUN})
                continue
            mean, low, high = paired_bootstrap(a, b, n_boot=n_boot)
            verdict = judge(low, high, float(claim.get("margin", 0.0)))
            rows.append({"dataset": ds, "diff": mean, "low": low, "high": high, "verdict": verdict})
        out.append({"claim": claim, "rows": rows, "overall": overall([r["verdict"] for r in rows])})
    return out


def render_claims_md(results: list[dict[str, Any]]) -> str:
    if not results:
        return ""
    parts = [
        "## Claims tested\n\n",
        "These claims were written in `configs/claims.yaml` and committed before the benchmark ran. "
        "Each number is the mean per-query difference (A minus B) with a 95% paired bootstrap interval "
        "over queries. `supported` means the whole interval clears the bar. This is evidence on these "
        "datasets and this one embedding model, not a general proof.\n\n",
    ]
    for r in results:
        c = r["claim"]
        parts.append(f"### {c['id']}: {c['text']}\n\n")
        margin = float(c.get("margin", 0.0))
        bar = f"A may be at most {margin} worse than B" if margin else "A must beat B"
        parts.append(
            f"A = `{c['a']['method']}` ({c['a']['mode']}), B = `{c['b']['method']}` ({c['b']['mode']}), "
            f"metric = {c['metric']}. Bar: {bar}. **Overall: {r['overall']}.**\n\n"
        )
        parts.append("| dataset | A minus B | 95% interval | verdict |\n|---|---|---|---|\n")
        for row in r["rows"]:
            if row["verdict"] == NOT_RUN:
                parts.append(f"| {row['dataset']} | — | — | {NOT_RUN} |\n")
            else:
                parts.append(
                    f"| {row['dataset']} | {row['diff']:+.4f} | "
                    f"[{row['low']:+.4f}, {row['high']:+.4f}] | {row['verdict']} |\n"
                )
        parts.append("\n")
    return "".join(parts)
