"""Static check and sandbox runner (hard rule 5)."""

import subprocess
import sys
from pathlib import Path

import pytest

from tests.test_evaluate import toy_prepared
from trivec.lab.runner import run_experiment
from trivec.lab.static_check import check_source, has_required_contract

GOOD_QUANTIZER = """
import numpy as np
from trivec.quantizers.base import Codes, FloatArray

class HalfTernary:
    name = "half_ternary"
    def __init__(self, alpha: float = 0.5) -> None:
        self.alpha = alpha
        self._dim = None
    def fit(self, corpus: FloatArray, *, seed: int) -> None:
        self._dim = corpus.shape[1]
    def encode(self, x: FloatArray) -> Codes:
        t = self.alpha * np.abs(x).mean(axis=1, keepdims=True)
        return np.where(np.abs(x) >= t, np.sign(x), 0).astype(np.int8)
    def decode(self, codes: Codes) -> FloatArray:
        return codes.astype(np.float32)
    def bytes_per_vector(self, dim: int) -> float:
        return float(dim)

def build(**params):
    return HalfTernary(**params)
"""

GOOD_TEST = """
import numpy as np
from quantizer import build

def test_shape():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((50, 16), dtype=np.float32)
    q = build(alpha=0.5); q.fit(x, seed=0)
    c = q.encode(x)
    assert c.shape == (50, 16) and c.nbytes / 50 == q.bytes_per_vector(16)
"""


@pytest.mark.parametrize(
    ("snippet", "needle"),
    [
        ("import socket\n", "socket"),
        ("from trivec.eval import evaluate\n", "trivec.eval"),
        ("import trivec.lab.runner\n", "trivec.lab"),
        ("x = open('f')\n", "open()"),
        ("import numpy as np\nnp.load('a.npy')\n", ".load"),
        ("exec('1')\n", "exec()"),
        ("import os\n", "'os'"),
        ("from . import x\n", "import from"),
        ("def f():\n    global g\n", "global"),
        ("import subprocess\n", "subprocess"),
        ("y = __builtins__\n", "__builtins__"),
    ],
)
def test_static_check_rejects(snippet: str, needle: str) -> None:
    problems = check_source(snippet)
    assert problems and any(needle in p for p in problems), problems


def test_static_check_accepts_good_code() -> None:
    assert check_source(GOOD_QUANTIZER) == []
    assert has_required_contract(GOOD_QUANTIZER) == []
    assert has_required_contract("import numpy\n") == [
        "missing module-level function build(**params)",
        "missing a quantizer class",
    ]


def test_network_guard_blocks_sockets() -> None:
    code = (
        "from trivec.lab import netguard; netguard.install(); import socket\n"
        "try:\n    socket.socket()\nexcept RuntimeError as e:\n    print('BLOCKED', e)\n"
    )
    src = str(Path(__file__).resolve().parents[1] / "src")
    proc = subprocess.run(
        [sys.executable, "-I", "-c", code],
        capture_output=True,
        text=True,
        env={"PYTHONPATH": src, "SYSTEMROOT": "C:\\Windows", "PATH": ""},
    )
    assert "BLOCKED" in proc.stdout, proc.stderr


def test_run_experiment_end_to_end(tmp_path: Path) -> None:
    exp = tmp_path / "exp_0001"
    exp.mkdir()
    (exp / "quantizer.py").write_text(GOOD_QUANTIZER, encoding="utf-8")
    (exp / "test_quantizer.py").write_text(GOOD_TEST, encoding="utf-8")
    out = run_experiment(
        exp,
        toy_prepared(n=200),
        quantizer_name="half_ternary",
        params={"alpha": 0.5},
        warmup=1,
        timeout_s=300,
        experiment_id="exp_0001",
    )
    assert out.ok, (
        out.message,
        out.tests_output,
        (exp / "run.log").read_text() if (exp / "run.log").exists() else "",
    )
    assert out.tests_passed is True
    rec = out.record or {}
    assert (
        rec["status"] == "ok"
        and rec["source"] == "lab"
        and rec["integrity"]["deterministic"] is True
    )
    assert rec["quantizer"] == "half_ternary" and 0 < rec["metrics"]["recall_at_10"] <= 1.0
    assert rec["metrics"]["bytes_per_vector"] == rec["metrics"]["bytes_per_vector_measured"] == 64.0


def test_run_experiment_rejects_bad_static(tmp_path: Path) -> None:
    exp = tmp_path / "exp_0002"
    exp.mkdir()
    (exp / "quantizer.py").write_text("import socket\n" + GOOD_QUANTIZER, encoding="utf-8")
    out = run_experiment(
        exp, toy_prepared(n=50), quantizer_name="half_ternary", params={}, warmup=1
    )
    assert not out.ok and out.stage == "static_check" and out.static_violations


def test_run_experiment_reports_failing_agent_tests(tmp_path: Path) -> None:
    exp = tmp_path / "exp_0003"
    exp.mkdir()
    (exp / "quantizer.py").write_text(GOOD_QUANTIZER, encoding="utf-8")
    (exp / "test_quantizer.py").write_text("def test_bad():\n    assert False\n", encoding="utf-8")
    out = run_experiment(
        exp, toy_prepared(n=50), quantizer_name="half_ternary", params={}, warmup=1, timeout_s=300
    )
    assert not out.ok and out.stage == "tests" and out.tests_passed is False


def test_run_experiment_records_runtime_failure(tmp_path: Path) -> None:
    exp = tmp_path / "exp_0004"
    exp.mkdir()
    bad = GOOD_QUANTIZER.replace("return codes.astype(np.float32)", "raise ValueError('boom')")
    (exp / "quantizer.py").write_text(bad, encoding="utf-8")
    out = run_experiment(
        exp, toy_prepared(n=50), quantizer_name="half_ternary", params={}, warmup=1, timeout_s=300
    )
    assert not out.ok and out.stage == "evaluate"
    assert (
        out.record is not None
        and out.record["status"] == "failed"
        and "boom" in out.record["error"]["message"]
    )
