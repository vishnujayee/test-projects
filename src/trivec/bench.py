"""`trivec bench`: run every quantizer x seed x search mode in a YAML config.

Config shape (configs/<name>.yaml):
    name: scifact_minilm
    dataset: scifact
    model: {name: sentence-transformers/all-MiniLM-L6-v2, revision: null}
    k: 10
    seeds: [0, 1, 2]
    warmup: 20
    quantizers:
      - {name: float32}
      - {name: ternary, params: {alpha: 0.7}, modes: [asymmetric_flat, symmetric_packed]}
"""

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from trivec.data.prepare import DEFAULT_CACHE_ROOT, PreparedData, prepare
from trivec.eval.evaluate import RunSpec, evaluate, write_result
from trivec.quantizers import make_quantizer

DEFAULT_RESULTS_DIR = Path("results")


@dataclass(frozen=True)
class BenchConfig:
    name: str
    dataset: str
    model_name: str
    model_revision: str | None
    k: int
    seeds: list[int]
    warmup: int
    quantizers: list[dict[str, Any]]
    raw: dict[str, Any]


def load_config(path: Path) -> BenchConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    model = raw["model"]
    return BenchConfig(
        name=str(raw["name"]),
        dataset=str(raw["dataset"]),
        model_name=str(model["name"]),
        model_revision=model.get("revision"),
        k=int(raw.get("k", 10)),
        seeds=[int(s) for s in raw.get("seeds", [0])],
        warmup=int(raw.get("warmup", 20)),
        quantizers=list(raw["quantizers"]),
        raw=raw,
    )


def iter_specs(cfg: BenchConfig) -> Iterator[RunSpec]:
    for entry in cfg.quantizers:
        name = str(entry["name"])
        params = dict(entry.get("params", {}))
        modes = list(entry.get("modes", ["asymmetric_flat"]))
        for mode in modes:
            for seed in cfg.seeds:
                yield RunSpec(
                    quantizer_name=name,
                    quantizer_params=params,
                    seed=seed,
                    k=cfg.k,
                    search_mode=str(mode),
                    warmup=cfg.warmup,
                    config=cfg.raw,
                )


def run_bench(
    config_path: Path,
    *,
    results_dir: Path = DEFAULT_RESULTS_DIR,
    cache_root: Path = DEFAULT_CACHE_ROOT,
    data: PreparedData | None = None,
) -> list[Path]:
    cfg = load_config(config_path)
    if data is None:
        data = prepare(
            cfg.dataset,
            cfg.model_name,
            model_revision=cfg.model_revision,
            k=cfg.k,
            cache_root=cache_root,
        )
    written: list[Path] = []
    for spec in iter_specs(cfg):
        quantizer = make_quantizer(spec.quantizer_name, **spec.quantizer_params)
        record = evaluate(spec, data, quantizer)
        path = write_result(record, results_dir)
        written.append(path)
        m = record["metrics"]
        if record["status"] == "ok":
            print(
                f"{spec.quantizer_name:8s} {spec.search_mode:16s} seed={spec.seed} "
                f"recall@{cfg.k}={m[f'recall_at_{cfg.k}']:.4f} ndcg@{cfg.k}={m[f'ndcg_at_{cfg.k}']:.4f} "
                f"bytes={m['bytes_per_vector']:.0f} p50={m['p50_latency_ms']:.2f}ms -> {path.name}"
            )
        else:
            print(
                f"{spec.quantizer_name:8s} {spec.search_mode:16s} seed={spec.seed} FAILED: {record['error']['message']}"
            )
    return written
