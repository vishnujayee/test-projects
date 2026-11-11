"""prepare(): dataset + model -> cached embeddings, qrels and ground truth.

Cache layout (docs/design.md section 6.3):
    data/cache/<dataset>@<zip_sha[:12]>/<model_slug>@<model_sha[:12]>/
        corpus.npy  queries.npy  corpus_ids.json  query_ids.json  qrels.json
        ground_truth_k<k>.npy  meta.json

Embedding happens once; every later experiment reuses the files.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from trivec.data.beir import Qrels, load_beir
from trivec.data.embed import embed_texts, model_slug, resolve_model_revision
from trivec.index.exact import exact_search
from trivec.quantizers.base import FloatArray, IntArray

DEFAULT_CACHE_ROOT = Path("data") / "cache"


@dataclass(frozen=True)
class PreparedData:
    dataset_name: str
    dataset_revision: str
    model_name: str
    model_revision: str
    corpus: FloatArray
    queries: FloatArray
    corpus_ids: list[str]
    query_ids: list[str]
    qrels: Qrels
    ground_truth: IntArray
    """Exact float32 top-k row indices into `corpus` for every query (hard rule 4)."""

    @property
    def dim(self) -> int:
        return int(self.corpus.shape[1])


def save_prepared(data: PreparedData, folder: Path) -> None:
    """Serialise for the sandbox subprocess (arrays as .npz, the rest as JSON)."""
    folder.mkdir(parents=True, exist_ok=True)
    np.savez(
        folder / "arrays.npz",
        corpus=data.corpus,
        queries=data.queries,
        ground_truth=data.ground_truth,
    )
    meta = {
        "dataset_name": data.dataset_name,
        "dataset_revision": data.dataset_revision,
        "model_name": data.model_name,
        "model_revision": data.model_revision,
        "corpus_ids": data.corpus_ids,
        "query_ids": data.query_ids,
        "qrels": data.qrels,
    }
    (folder / "meta.json").write_text(json.dumps(meta), encoding="utf-8")


def load_prepared(folder: Path) -> PreparedData:
    arrays = np.load(folder / "arrays.npz")
    meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
    return PreparedData(
        dataset_name=meta["dataset_name"],
        dataset_revision=meta["dataset_revision"],
        model_name=meta["model_name"],
        model_revision=meta["model_revision"],
        corpus=arrays["corpus"],
        queries=arrays["queries"],
        corpus_ids=list(meta["corpus_ids"]),
        query_ids=list(meta["query_ids"]),
        qrels={q: {d: int(g) for d, g in rel.items()} for q, rel in meta["qrels"].items()},
        ground_truth=arrays["ground_truth"],
    )


def cache_dir(cache_root: Path, dataset: str, dataset_rev: str, model: str, model_rev: str) -> Path:
    return cache_root / f"{dataset}@{dataset_rev[:12]}" / f"{model_slug(model)}@{model_rev[:12]}"


def prepare(
    dataset: str,
    model: str,
    *,
    model_revision: str | None = None,
    k: int = 10,
    cache_root: Path = DEFAULT_CACHE_ROOT,
) -> PreparedData:
    beir = load_beir(dataset, cache_root)
    model_rev = resolve_model_revision(model, model_revision)
    folder = cache_dir(cache_root, dataset, beir.revision, model, model_rev)
    folder.mkdir(parents=True, exist_ok=True)

    corpus_path, queries_path = folder / "corpus.npy", folder / "queries.npy"
    if not corpus_path.exists():
        np.save(folder / "corpus.tmp.npy", embed_texts(beir.corpus_texts, model, model_rev))
        (folder / "corpus.tmp.npy").replace(corpus_path)
    if not queries_path.exists():
        np.save(folder / "queries.tmp.npy", embed_texts(beir.query_texts, model, model_rev))
        (folder / "queries.tmp.npy").replace(queries_path)
    corpus: FloatArray = np.load(corpus_path)
    queries: FloatArray = np.load(queries_path)

    (folder / "corpus_ids.json").write_text(json.dumps(beir.corpus_ids), encoding="utf-8")
    (folder / "query_ids.json").write_text(json.dumps(beir.query_ids), encoding="utf-8")
    (folder / "qrels.json").write_text(json.dumps(beir.qrels), encoding="utf-8")

    gt_path = folder / f"ground_truth_k{k}.npy"
    if not gt_path.exists():
        ids, _ = exact_search(queries, corpus, k)
        np.save(gt_path, ids)
    ground_truth: IntArray = np.load(gt_path)

    meta = {
        "dataset": dataset,
        "dataset_revision": beir.revision,
        "model": model,
        "model_revision": model_rev,
        "n_corpus": len(beir.corpus_ids),
        "n_queries": len(beir.query_ids),
        "dim": int(corpus.shape[1]),
    }
    (folder / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return PreparedData(
        dataset_name=dataset,
        dataset_revision=beir.revision,
        model_name=model,
        model_revision=model_rev,
        corpus=corpus,
        queries=queries,
        corpus_ids=beir.corpus_ids,
        query_ids=beir.query_ids,
        qrels=beir.qrels,
        ground_truth=ground_truth,
    )
