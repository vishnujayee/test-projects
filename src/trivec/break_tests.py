"""`trivec break`: try to break the benchmark, and save what happened.

CLAUDE.md: "If a result looks too good, try to break it before reporting it as a win."
Each check has a prediction that would expose a bug or an artefact if it failed. The output is
results/checks/break_tests.json, and docs/results.md renders it, so no number is typed by hand.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np

from trivec.bench import load_config
from trivec.data.prepare import PreparedData, prepare
from trivec.eval.metrics import ndcg_at_k, recall_per_query
from trivec.eval.provenance import collect
from trivec.index.exact import exact_search
from trivec.index.packed import PackedPopcountIndex
from trivec.quantizers import BinaryQuantizer, Int8Quantizer, TernaryQuantizer
from trivec.quantizers.base import FloatArray, IntArray

K = 10
ALPHA = 0.7


def _recall(ids: IntArray, truth: IntArray) -> float:
    return float(np.mean(recall_per_query(ids, truth[:, :K])))


def _decoded(corpus: FloatArray) -> dict[str, FloatArray]:
    out: dict[str, FloatArray] = {}
    for name, q in (
        ("int8", Int8Quantizer()),
        ("binary", BinaryQuantizer()),
        ("ternary", TernaryQuantizer(alpha=ALPHA)),
    ):
        q.fit(corpus, seed=0)
        out[name] = q.decode(q.encode(corpus))
    return out


def shuffled_labels(data: PreparedData) -> dict[str, float]:
    """Give every query another query's labels. nDCG must collapse, or the pipeline leaks."""
    ids, _ = exact_search(data.queries, data.corpus, K + 1)
    real = ndcg_at_k(ids, data.query_ids, data.corpus_ids, data.qrels, K)
    perm = np.random.default_rng(0).permutation(len(data.query_ids))
    shuffled = {
        qid: data.qrels.get(data.query_ids[int(p)], {})
        for qid, p in zip(data.query_ids, perm, strict=True)
    }
    fake = ndcg_at_k(ids, data.query_ids, data.corpus_ids, shuffled, K)
    return {"real_ndcg": real, "shuffled_ndcg": fake}


def recall_without_self(data: PreparedData) -> dict[str, Any]:
    """In ArguAna most queries are also documents, which may hand out a free neighbour."""
    position = {d: i for i, d in enumerate(data.corpus_ids)}
    self_rows = np.array([position.get(q, -1) for q in data.query_ids])
    truth_all, _ = exact_search(data.queries, data.corpus, K + 1)

    def strip(ids: IntArray) -> IntArray:
        return np.array(
            [[i for i in row if i != s][:K] for row, s in zip(ids, self_rows, strict=True)]
        )

    truth = strip(truth_all)
    in_corpus = self_rows >= 0
    top1_self = (
        float(np.mean(truth_all[in_corpus, 0] == self_rows[in_corpus])) if in_corpus.any() else 0.0
    )
    out: dict[str, Any] = {
        "queries_also_in_corpus": int(in_corpus.sum()),
        "n_queries": len(data.query_ids),
        "self_is_top1_among_those": top1_self,
    }
    for name, decoded in _decoded(data.corpus).items():
        ids, _ = exact_search(data.queries, decoded, K + 1)
        out[name] = {
            "with_self": _recall(ids, truth_all),
            "without_self": _recall(strip(ids), truth),
        }
    return out


def rotation(data: PreparedData, n_rotations: int = 3) -> dict[str, dict[str, float]]:
    """A random rotation keeps every dot product. Exact search must not change, but methods that
    trust individual coordinates may: this shows whether a win depends on the embedding's axes."""
    dim = data.corpus.shape[1]
    out: dict[str, list[float]] = {"float32": [], "binary": [], "ternary": []}
    for seed in range(n_rotations):
        rng = np.random.default_rng(seed)
        rot, _ = np.linalg.qr(rng.standard_normal((dim, dim)))
        rot = rot.astype(np.float32)
        corpus = np.ascontiguousarray(data.corpus @ rot, dtype=np.float32)
        queries = np.ascontiguousarray(data.queries @ rot, dtype=np.float32)
        ids, _ = exact_search(queries, corpus, K)
        out["float32"].append(_recall(ids, data.ground_truth))
        decoded = _decoded(corpus)
        for name in ("binary", "ternary"):
            ids, _ = exact_search(queries, decoded[name], K)
            out[name].append(_recall(ids, data.ground_truth))
    original = _decoded(data.corpus)
    baseline = {
        name: _recall(exact_search(data.queries, original[name], K)[0], data.ground_truth)
        for name in ("binary", "ternary")
    }
    baseline["float32"] = 1.0
    return {
        name: {"original": baseline[name], "rotated_mean": float(np.mean(vals))}
        for name, vals in out.items()
    }


def tie_order(data: PreparedData) -> dict[str, dict[str, float]]:
    """Packed scores are small integers, so many documents tie. If tie order moved recall much,
    the packed numbers would be an accident of document order. Reverse the corpus and compare."""
    out: dict[str, dict[str, float]] = {}
    n = data.corpus.shape[0]
    for name, make in (("binary", BinaryQuantizer), ("ternary", lambda: TernaryQuantizer(ALPHA))):
        q = make()
        q.fit(data.corpus, seed=0)
        forward, _ = PackedPopcountIndex(q, q.encode(data.corpus)).search(data.queries, K)
        reverse_corpus = np.ascontiguousarray(data.corpus[::-1])
        q2 = make()
        q2.fit(reverse_corpus, seed=0)
        back, _ = PackedPopcountIndex(q2, q2.encode(reverse_corpus)).search(data.queries, K)
        out[name] = {
            "forward": _recall(forward, data.ground_truth),
            "reversed": _recall(n - 1 - back, data.ground_truth),
        }
    return out


def scale_variants(data: PreparedData) -> dict[str, dict[str, float]]:
    """Ternary at alpha=0 should match binary but scored lower, which points at the per-vector
    scale. Compare our least-squares scale with no scale at all (which would also save 4 bytes)
    and with a scale that restores unit length."""
    q = TernaryQuantizer(alpha=ALPHA)
    q.fit(data.corpus, seed=0)
    codes, scale = q.ternarize(data.corpus)
    signs = codes.astype(np.float32)
    nnz = np.maximum((codes != 0).sum(axis=1), 1).astype(np.float32)
    variants = {
        "least_squares_scale": signs * scale[:, None],
        "no_scale": signs,
        "unit_length": signs / np.sqrt(nnz)[:, None],
    }
    out: dict[str, dict[str, float]] = {}
    for name, decoded in variants.items():
        ids, _ = exact_search(data.queries, np.ascontiguousarray(decoded), K + 1)
        out[name] = {
            "recall": _recall(ids, data.ground_truth),
            "ndcg": ndcg_at_k(ids, data.query_ids, data.corpus_ids, data.qrels, K),
        }
    return out


def threshold_theory(data: PreparedData) -> dict[str, float]:
    """Is alpha near 0.7 luck? The classic best 3-level quantizer (Lloyd's algorithm) can be solved
    directly on the real values: threshold = half of the mean kept magnitude, repeated."""
    a = np.abs(data.corpus)
    means = a.mean(axis=1, keepdims=True)
    normalised = (a / means).ravel()
    t = 1.0
    for _ in range(100):
        t = float(normalised[normalised >= t].mean()) / 2.0
    gaussian = np.abs(np.random.default_rng(0).standard_normal(2_000_000))
    gaussian /= gaussian.mean()
    g = 1.0
    for _ in range(100):
        g = float(gaussian[gaussian >= g].mean()) / 2.0
    q = TernaryQuantizer(alpha=ALPHA)
    codes, _ = q.ternarize(data.corpus)
    values, counts = np.unique(codes, return_counts=True)
    p = counts / counts.sum()
    return {
        "lloyd_alpha_on_this_data": t,
        "lloyd_alpha_gaussian": g,
        "zero_fraction_at_alpha_0.7": float(np.mean(codes == 0)),
        "bits_of_information_per_dim": float(-(p * np.log2(p)).sum()),
        "n_levels": float(len(values)),
    }


def run_break_tests(config_paths: list[Path], out_path: Path) -> dict[str, Any]:
    report: dict[str, Any] = {"provenance": collect(), "alpha": ALPHA, "k": K, "datasets": {}}
    for path in config_paths:
        cfg = load_config(path)
        data = prepare(cfg.dataset, cfg.model_name, model_revision=cfg.model_revision, k=cfg.k)
        report["datasets"][cfg.dataset] = {
            "model": cfg.model_name,
            "shuffled_labels": shuffled_labels(data),
            "recall_without_self": recall_without_self(data),
            "rotation": rotation(data),
            "tie_order": tie_order(data),
            "scale_variants": scale_variants(data),
            "threshold_theory": threshold_theory(data),
        }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(report, indent=2), encoding="utf-8")
    tmp.replace(out_path)
    return report


def render_break_md(report: dict[str, Any]) -> str:
    """Tables for docs/results.md, built only from the saved JSON."""
    ds = report.get("datasets") or {}
    if not ds:
        return ""
    names = sorted(ds)
    parts = [
        "## Attempts to break these results\n\n",
        "Generated by `uv run trivec break`, saved in `results/checks/break_tests.json`. "
        "Each check states what a bug or an artefact would look like.\n\n",
        "**Shuffled labels.** Give each query another query's labels. nDCG must collapse, or the pipeline leaks.\n\n",
        "| dataset | real nDCG@10 | shuffled nDCG@10 |\n|---|---|---|\n",
    ]
    for n in names:
        s = ds[n]["shuffled_labels"]
        parts.append(f"| {n} | {s['real_ndcg']:.4f} | {s['shuffled_ndcg']:.4f} |\n")
    parts.append(
        "\n**Queries that are also documents.** A query that finds itself gets a free neighbour. "
        "Recall with and without that hit:\n\n"
        "| dataset | queries also in corpus | binary with / without | ternary with / without | int8 with / without |\n"
        "|---|---|---|---|---|\n"
    )
    for n in names:
        r = ds[n]["recall_without_self"]
        cells = " | ".join(
            f"{r[m]['with_self']:.4f} / {r[m]['without_self']:.4f}"
            for m in ("binary", "ternary", "int8")
        )
        parts.append(f"| {n} | {r['queries_also_in_corpus']} of {r['n_queries']} | {cells} |\n")
    parts.append(
        "\n**Random rotation.** Rotating every vector keeps all dot products, so float32 must stay at 1.0. "
        "Binary and ternary trust individual coordinates, so this shows whether they depend on the model's axes.\n\n"
        "| dataset | float32 rotated | binary original / rotated | ternary original / rotated |\n|---|---|---|---|\n"
    )
    for n in names:
        r = ds[n]["rotation"]
        parts.append(
            f"| {n} | {r['float32']['rotated_mean']:.4f} | "
            f"{r['binary']['original']:.4f} / {r['binary']['rotated_mean']:.4f} | "
            f"{r['ternary']['original']:.4f} / {r['ternary']['rotated_mean']:.4f} |\n"
        )
    parts.append(
        "\n**Tie order.** Packed scores tie often. Reversing the document order shows how much recall depends on tie-breaking.\n\n"
        "| dataset | binary forward / reversed | ternary forward / reversed |\n|---|---|---|\n"
    )
    for n in names:
        r = ds[n]["tie_order"]
        parts.append(
            f"| {n} | {r['binary']['forward']:.4f} / {r['binary']['reversed']:.4f} | "
            f"{r['ternary']['forward']:.4f} / {r['ternary']['reversed']:.4f} |\n"
        )
    parts.append(
        "\n**What the per-vector scale does.** Ternary (alpha=0.7) with three ways to set the length of the rebuilt vector, "
        "shown as recall / nDCG. This is exploratory: it was not a pre-registered claim.\n\n"
        "| dataset | least-squares scale (ours, 100 B) | no scale (96 B) | unit length (needs no stored scale, 96 B) |\n|---|---|---|---|\n"
    )
    for n in names:
        r = ds[n]["scale_variants"]
        cells = " | ".join(
            f"{r[m]['recall']:.4f} / {r[m]['ndcg']:.4f}"
            for m in ("least_squares_scale", "no_scale", "unit_length")
        )
        parts.append(f"| {n} | {cells} |\n")
    parts.append(
        "\n**Is alpha near 0.7 luck?** Lloyd's algorithm finds the best 3-level quantizer directly from the values, "
        "with no search over recall.\n\n"
        "| dataset | Lloyd alpha on this data | Lloyd alpha for a Gaussian | zeros at alpha=0.7 | information per dimension (bits) |\n|---|---|---|---|---|\n"
    )
    for n in names:
        r = ds[n]["threshold_theory"]
        parts.append(
            f"| {n} | {r['lloyd_alpha_on_this_data']:.3f} | {r['lloyd_alpha_gaussian']:.3f} | "
            f"{r['zero_fraction_at_alpha_0.7']:.3f} | {r['bits_of_information_per_dim']:.3f} |\n"
        )
    parts.append("\n")
    return "".join(parts)
