"""Lab configuration (configs/lab.yaml)."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from trivec.lab.llm import LLMConfig


@dataclass(frozen=True)
class LabPaths:
    lab_dir: Path = Path("lab")
    results_dir: Path = Path("results")
    reports_dir: Path = Path("reports")
    docs_dir: Path = Path("docs")

    @property
    def state_file(self) -> Path:
        return self.lab_dir / "state.json"

    @property
    def experiments_dir(self) -> Path:
        return self.lab_dir / "experiments"

    @property
    def library_dir(self) -> Path:
        return self.lab_dir / "library"


@dataclass(frozen=True)
class LabConfig:
    llm: LLMConfig = field(default_factory=LLMConfig)
    bench_config: Path = Path("configs/scifact_minilm.yaml")
    holdout_configs: list[Path] = field(
        default_factory=lambda: [Path("configs/nfcorpus_minilm.yaml")]
    )
    sandbox_timeout_s: float = 1200.0
    seed: int = 0
    search_mode: str = "asymmetric_flat"
    librarian_enabled: bool = True
    librarian_queries: list[str] = field(
        default_factory=lambda: [
            "ternary quantization embeddings retrieval",
            "binary embeddings similarity search",
            "product quantization nearest neighbor search",
            "vector compression dense retrieval",
        ]
    )
    librarian_max_per_query: int = 8
    librarian_embedder: str = "hash"  # "hash" (offline) | a sentence-transformers model name
    paths: LabPaths = field(default_factory=LabPaths)
    raw: dict[str, Any] = field(default_factory=dict)


def load_lab_config(path: Path) -> LabConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    llm_raw = dict(raw.get("llm", {}))
    llm = LLMConfig(**{k: v for k, v in llm_raw.items() if k in LLMConfig.__dataclass_fields__})
    lib = dict(raw.get("librarian", {}))
    paths_raw = dict(raw.get("paths", {}))
    paths = LabPaths(
        **{k: Path(v) for k, v in paths_raw.items() if k in LabPaths.__dataclass_fields__}
    )
    sandbox = dict(raw.get("sandbox", {}))
    return LabConfig(
        llm=llm,
        bench_config=Path(raw.get("bench_config", "configs/scifact_minilm.yaml")),
        holdout_configs=[
            Path(p) for p in raw.get("holdout_configs", ["configs/nfcorpus_minilm.yaml"])
        ],
        sandbox_timeout_s=float(sandbox.get("timeout_s", 1200.0)),
        seed=int(raw.get("seed", 0)),
        search_mode=str(raw.get("search_mode", "asymmetric_flat")),
        librarian_enabled=bool(lib.get("enabled", True)),
        librarian_queries=list(lib.get("queries", LabConfig().librarian_queries)),
        librarian_max_per_query=int(lib.get("max_papers_per_query", 8)),
        librarian_embedder=str(lib.get("embedder", "hash")),
        paths=paths,
        raw=raw,
    )
