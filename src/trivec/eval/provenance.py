"""Provenance: everything needed to re-run a result (hard rule 2)."""

import datetime as dt
import functools
import os
import platform
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from typing import Any

TRACKED_PACKAGES = ("numpy", "faiss-cpu", "sentence-transformers", "torch", "trivec")
CODE_PATHS = (":/src", ":/configs", ":/pyproject.toml", ":/uv.lock")


def git_info(repo: Path | None = None) -> dict[str, Any]:
    cwd = str(repo) if repo else None
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True, text=True, check=True
        ).stdout.strip()
        # code only: result files written by the bench itself must not count as "dirty".
        # ":/" anchors paths at the repo root, so this also works from lab/experiments/<id>/.
        status = subprocess.run(
            ["git", "status", "--porcelain", "--", *CODE_PATHS],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        return {"git_commit": commit, "git_dirty": bool(status.strip())}
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"git_commit": None, "git_dirty": None}


def package_versions() -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for name in TRACKED_PACKAGES:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = None
    return out


@functools.lru_cache(maxsize=1)
def cpu_name() -> str:
    try:
        if sys.platform.startswith("linux"):
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
        if sys.platform == "win32":
            r = subprocess.run(
                ["wmic", "cpu", "get", "name"], capture_output=True, text=True, check=False
            )
            lines = [ln.strip() for ln in r.stdout.splitlines() if ln.strip()]
            if len(lines) >= 2:
                return lines[1]
    except OSError:
        pass
    return platform.processor() or platform.machine()


def hardware() -> dict[str, Any]:
    ram_gb: float | None
    try:
        import psutil

        ram_gb = round(psutil.virtual_memory().total / 1e9, 2)
    except ImportError:
        ram_gb = None
    return {
        "platform": platform.platform(),
        "cpu": cpu_name(),
        "cpu_count": os.cpu_count(),
        "ram_gb": ram_gb,
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
    }


def collect(repo: Path | None = None) -> dict[str, Any]:
    return {
        **git_info(repo),
        "packages": package_versions(),
        "hardware": hardware(),
        "python": platform.python_version(),
        "timestamp": dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
