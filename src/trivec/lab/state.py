"""Lab state on disk: atomic JSON writes so a crash never leaves half a file."""

import datetime as dt
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

STAGES = ("librarian", "proposer", "coder", "runner", "critic", "logger", "done")


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def atomic_write_json(path: Path, obj: Any) -> None:
    atomic_write_text(path, json.dumps(obj, indent=2, default=str))


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def now_iso() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass
class LabState:
    experiment_id: str
    stage: str
    started_at: str
    updated_at: str

    @classmethod
    def load(cls, path: Path) -> "LabState | None":
        data = read_json(path)
        return cls(**data) if data else None

    def save(self, path: Path) -> None:
        self.updated_at = now_iso()
        atomic_write_json(path, asdict(self))

    def advance(self, path: Path) -> None:
        i = STAGES.index(self.stage)
        self.stage = STAGES[min(i + 1, len(STAGES) - 1)]
        self.save(path)


def next_experiment_id(experiments_dir: Path) -> str:
    experiments_dir.mkdir(parents=True, exist_ok=True)
    existing = [
        p.name for p in experiments_dir.iterdir() if p.is_dir() and p.name.startswith("exp_")
    ]
    n = max((int(name.split("_")[1]) for name in existing), default=0) + 1
    return f"exp_{n:04d}"
