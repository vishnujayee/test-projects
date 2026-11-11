"""Try to cheat the Lab on purpose. Each quantizer here would post wonderful numbers if the
checks did not catch it, and each passes the static check, so only the runtime checks stop it.
"""

from pathlib import Path

import pytest

from tests.test_evaluate import toy_prepared
from tests.test_lab_sandbox import GOOD_QUANTIZER
from trivec.eval.memory import deep_size
from trivec.lab.critic import rule_based
from trivec.lab.runner import RunOutcome, run_experiment
from trivec.lab.static_check import check_source

# Stores the whole corpus in self. Claims 2 bytes per vector with perfect recall.
MEMORIZER = """
import numpy as np

class Memorizer:
    name = "memorizer"
    def __init__(self):
        self.vectors = None
    def fit(self, corpus, *, seed):
        self.vectors = np.array(corpus)
    def encode(self, x):
        return np.arange(x.shape[0], dtype=np.uint16).reshape(-1, 1)
    def decode(self, codes):
        return self.vectors[codes[:, 0].astype(np.int64)]
    def bytes_per_vector(self, dim):
        return 2.0

def build(**params):
    return Memorizer()
"""

# Same trick, hiding the corpus in a module-level dict instead of the instance.
GLOBAL_STASH = """
import numpy as np

STASH = {}

class Stasher:
    name = "stasher"
    def fit(self, corpus, *, seed):
        STASH["vectors"] = np.array(corpus)
    def encode(self, x):
        return np.arange(x.shape[0], dtype=np.uint16).reshape(-1, 1)
    def decode(self, codes):
        return STASH["vectors"][codes[:, 0].astype(np.int64)]
    def bytes_per_vector(self, dim):
        return 2.0

def build(**params):
    return Stasher()
"""

# Decode ignores its input and always returns the stored corpus.
IGNORES_CODES = """
import numpy as np

class Ignores:
    name = "ignores"
    def __init__(self):
        self.vectors = None
    def fit(self, corpus, *, seed):
        self.vectors = np.array(corpus)
    def encode(self, x):
        return np.zeros((x.shape[0], 1), dtype=np.uint8)
    def decode(self, codes):
        return self.vectors[: codes.shape[0]]
    def bytes_per_vector(self, dim):
        return 1.0

def build(**params):
    return Ignores()
"""

# Different codes on every build: results could never be reproduced.
RANDOM_CODES = """
import numpy as np

COUNTER = [0]

class Unstable:
    name = "unstable"
    def fit(self, corpus, *, seed):
        COUNTER[0] += 1
    def encode(self, x):
        return (x > (COUNTER[0] * 0.01)).astype(np.int8)
    def decode(self, codes):
        return codes.astype(np.float32)
    def bytes_per_vector(self, dim):
        return float(dim)

def build(**params):
    return Unstable()
"""


def _run(tmp_path: Path, source: str, name: str) -> RunOutcome:
    exp = tmp_path / name
    exp.mkdir()
    (exp / "quantizer.py").write_text(source, encoding="utf-8")
    return run_experiment(
        exp, toy_prepared(n=200), quantizer_name=name, params={}, warmup=1, timeout_s=300
    )


@pytest.mark.parametrize(
    ("source", "name", "flag"),
    [
        (MEMORIZER, "memorizer", "state_growth_ok"),
        (GLOBAL_STASH, "stasher", "state_growth_ok"),
        (IGNORES_CODES, "ignores", "decode_follows_codes"),
        (RANDOM_CODES, "unstable", "deterministic"),
    ],
)
def test_cheating_quantizer_is_caught(tmp_path: Path, source: str, name: str, flag: str) -> None:
    assert check_source(source) == [], "these must pass the static check to test the runtime checks"
    out = _run(tmp_path, source, name)
    assert out.record is not None and out.record["status"] == "ok", out.record
    assert out.record["integrity"][flag] is False, out.record["integrity"]
    critique = rule_based(out, tmp_path)
    assert critique.verdict == "reject", critique


def test_cheater_would_have_looked_excellent(tmp_path: Path) -> None:
    """The point of the checks: without them, these numbers would top a leaderboard."""
    out = _run(tmp_path, MEMORIZER, "memorizer")
    metrics = (out.record or {})["metrics"]
    assert metrics["recall_at_10"] == 1.0 and metrics["bytes_per_vector"] == 2.0
    # and honest accounting exposes it even before the integrity flag
    assert metrics["bytes_per_vector_amortized"] > 200


def test_honest_quantizer_passes_every_check(tmp_path: Path) -> None:
    source = GOOD_QUANTIZER.replace('name = "half_ternary"', 'name = "honest"')
    integrity = (_run(tmp_path, source, "honest").record or {})["integrity"]
    assert integrity["deterministic"] and integrity["decode_follows_codes"]
    assert integrity["state_growth_ok"]


def test_tampering_during_a_run_is_caught(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"n": 0}

    def fake_check() -> list[str]:
        calls["n"] += 1
        return [] if calls["n"] == 1 else ["modified: metrics.py"]

    monkeypatch.setattr("trivec.lab.runner.frozen.check", fake_check)
    out = _run(tmp_path, GOOD_QUANTIZER, "half_ternary")
    assert out.stage == "tampered" and not out.ok
    assert rule_based(out, tmp_path).verdict == "tampered"


@pytest.mark.parametrize(
    "snippet",
    [
        "getattr(object, '__glo' + 'bals__')\n",
        "from numpy import ctypeslib\n",
        "import numpy.ctypeslib\n",
        "from numpy import *\n",
        "x = ().__class__\n",
    ],
)
def test_static_check_blocks_evasions(snippet: str) -> None:
    assert check_source(snippet)


def test_deep_size_follows_containers_and_attributes() -> None:
    class Holder:
        def __init__(self) -> None:
            self.items = [b"abcd", {"k": b"123456"}]

    assert deep_size(Holder()) == 5 + 4 + 1 + 6  # keys "items" and "k" count too
