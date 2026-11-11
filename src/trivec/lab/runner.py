"""The Runner: executes agent code in a sandboxed subprocess (hard rule 5).

Layers (docs/design.md 7.3): static check -> `python -I` subprocess with timeout and the
network guard -> evaluator hash check before and after. Every outcome is recorded.
"""

import json
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from trivec.data.prepare import PreparedData, save_prepared
from trivec.eval import frozen
from trivec.lab.state import atomic_write_json
from trivec.lab.static_check import check_file, check_test_file, has_required_contract


@dataclass
class RunOutcome:
    ok: bool
    stage: str  # "static_check" | "tests" | "evaluate" | "tampered"
    static_violations: list[str] = field(default_factory=list)
    tests_passed: bool | None = None
    tests_output: str = ""
    record: dict[str, Any] | None = None
    eval_problems_before: list[str] = field(default_factory=list)
    eval_problems_after: list[str] = field(default_factory=list)
    message: str = ""


def _base_env() -> dict[str, str]:
    import os

    env = {
        k: v
        for k, v in os.environ.items()
        if k.upper() in {"PATH", "SYSTEMROOT", "TEMP", "TMP", "HOME", "USERPROFILE"}
    }
    env.update(
        {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "TRIVEC_SANDBOX": "1",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
        }
    )
    return env


def _python_cmd() -> list[str]:
    # -I ignores PYTHON* variables and the current directory; trivec comes from the venv install.
    # -B writes no .pyc files next to agent code.
    cmd = [sys.executable, "-I", "-B"]
    if sys.platform.startswith("linux") and shutil.which("unshare"):
        return ["unshare", "-n", *cmd]  # true network namespace on Linux CI
    return cmd


def run_agent_tests(test_path: Path, exp_dir: Path, timeout_s: float) -> tuple[bool, str]:
    cmd = [*_python_cmd(), "-m", "pytest", "-q", "-p", "trivec.lab.netguard", str(test_path)]
    try:
        proc = subprocess.run(
            cmd, cwd=exp_dir, env=_base_env(), capture_output=True, text=True, timeout=timeout_s
        )
    except subprocess.TimeoutExpired:
        return False, f"tests timed out after {timeout_s}s"
    return proc.returncode == 0, (proc.stdout + proc.stderr)[-6000:]


def run_experiment(
    exp_dir: Path,
    data: PreparedData,
    *,
    quantizer_name: str,
    params: dict[str, Any],
    seed: int = 0,
    k: int = 10,
    search_mode: str = "asymmetric_flat",
    warmup: int = 20,
    timeout_s: float = 1200.0,
    config: dict[str, Any] | None = None,
    experiment_id: str | None = None,
) -> RunOutcome:
    quantizer_path = exp_dir / "quantizer.py"
    test_path = exp_dir / "test_quantizer.py"

    violations = check_file(quantizer_path) + has_required_contract(
        quantizer_path.read_text(encoding="utf-8")
    )
    if test_path.exists():
        violations += [f"test: {v}" for v in check_test_file(test_path)]
    if violations:
        return RunOutcome(
            False, "static_check", static_violations=violations, message="rejected by static check"
        )

    before = frozen.check()
    if before:
        return RunOutcome(
            False, "tampered", eval_problems_before=before, message="evaluator modified before run"
        )

    tests_passed, tests_output = (None, "")
    if test_path.exists():
        tests_passed, tests_output = run_agent_tests(test_path, exp_dir, timeout_s)
        if not tests_passed:
            return RunOutcome(
                False,
                "tests",
                tests_passed=False,
                tests_output=tests_output,
                message="agent tests failed",
            )

    data_dir = exp_dir / "data"
    save_prepared(data, data_dir)
    job = {
        "quantizer_path": str(quantizer_path),
        "quantizer_name": quantizer_name,
        "params": params,
        "data_dir": str(data_dir),
        "out_json": str(exp_dir / "result.json"),
        "seed": seed,
        "k": k,
        "search_mode": search_mode,
        "warmup": warmup,
        "config": config or {},
        "experiment_id": experiment_id,
    }
    job_path = exp_dir / "job.json"
    atomic_write_json(job_path, job)
    # a result left over from an earlier run must never be mistaken for this run's result
    (exp_dir / "result.json").unlink(missing_ok=True)

    cmd = [*_python_cmd(), "-m", "trivec.lab.sandbox_entry", str(job_path)]
    try:
        proc = subprocess.run(
            cmd, cwd=exp_dir, env=_base_env(), capture_output=True, text=True, timeout=timeout_s
        )
        log = (proc.stdout + proc.stderr)[-6000:]
    except subprocess.TimeoutExpired:
        (exp_dir / "run.log").write_text(f"timed out after {timeout_s}s\n", encoding="utf-8")
        return RunOutcome(
            False,
            "evaluate",
            tests_passed=tests_passed,
            tests_output=tests_output,
            message="timeout",
        )
    (exp_dir / "run.log").write_text(log, encoding="utf-8")

    after = frozen.check()
    if after:
        return RunOutcome(
            False, "tampered", eval_problems_after=after, message="evaluator modified during run"
        )

    out = exp_dir / "result.json"
    if not out.exists():
        return RunOutcome(
            False,
            "evaluate",
            tests_passed=tests_passed,
            tests_output=tests_output,
            message=f"no result written; exit {proc.returncode}: {log[-500:]}",
        )
    record = json.loads(out.read_text(encoding="utf-8"))
    ok = record.get("status") == "ok"
    return RunOutcome(
        ok,
        "evaluate",
        tests_passed=tests_passed,
        tests_output=tests_output,
        record=record,
        message="ok" if ok else "evaluation failed",
    )
