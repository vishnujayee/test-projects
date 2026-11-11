"""Librarian, proposer/coder parsing, critic rules, and the full loop with a FakeLLM (resumable)."""

import json
from pathlib import Path

import pytest

from tests.test_evaluate import toy_prepared
from tests.test_lab_sandbox import GOOD_QUANTIZER, GOOD_TEST
from trivec.bench import run_bench
from trivec.lab import library as lib
from trivec.lab.agents import parse_code, parse_spec
from trivec.lab.config import LabConfig, LabPaths
from trivec.lab.critic import rule_based
from trivec.lab.llm import FakeLLM, LLMError
from trivec.lab.loop import Lab, approve, lab_status
from trivec.lab.runner import RunOutcome
from trivec.lab.state import LabState
from trivec.report import load_verification

PROPOSAL = json.dumps(
    {
        "title": "Half-threshold ternary",
        "hypothesis": "a lower alpha keeps more signal",
        "quantizer_name": "half_ternary",
        "method": "ternarize with alpha 0.5, no scale",
        "params": {"alpha": 0.5},
        "expected_bytes_per_vector": "d",
        "risks": "none",
    }
)
CODE = json.dumps({"quantizer_py": GOOD_QUANTIZER, "test_py": GOOD_TEST})
REVIEW = json.dumps(
    {"implements_spec": True, "leakage_risk": "low", "concerns": [], "summary": "fine"}
)


def fake_papers(query: str, n: int) -> list[lib.Paper]:
    return [
        lib.Paper(
            "p1",
            "Ternary quantization of embeddings",
            "We compress vectors with ternary codes for retrieval.",
            2024,
            "u1",
            "arxiv",
        ),
        lib.Paper(
            "p2",
            "Binary hashing for similarity search",
            "Binary embedding codes with hashing for nearest neighbor search.",
            2023,
            "u2",
            "arxiv",
        ),
        lib.Paper("p3", "A paper about cooking", "Recipes and kitchens.", 2020, "u3", "arxiv"),
    ]


def test_library_filters_topic_and_searches(tmp_path: Path) -> None:
    library = lib.Library(tmp_path / "library")
    added = library.collect(["q"], 5, [fake_papers])
    assert added == 2 and len(library.notes()) == 2  # the cooking paper is off-topic
    assert library.collect(["q"], 5, [fake_papers]) == 0  # idempotent
    hits = library.search("ternary codes embeddings", lib.hash_embedder(), k=1)
    assert hits and hits[0][0] == "p1"


def test_parse_spec_and_code() -> None:
    spec = parse_spec(PROPOSAL, ["ternary"])
    assert spec.quantizer_name == "half_ternary" and spec.params == {"alpha": 0.5}
    with pytest.raises(LLMError):
        parse_spec(PROPOSAL, ["half_ternary"])
    q, t = parse_code(CODE)
    assert "def build" in q and "def test_shape" in t


def test_critic_rules(tmp_path: Path) -> None:
    results = tmp_path / "results"
    results.mkdir()
    rec = {
        "status": "ok",
        "k": 10,
        "search_mode": "asymmetric_flat",
        "dataset": {"name": "toy"},
        "integrity": {"deterministic": True},
        "metrics": {
            "recall_at_10": 0.5,
            "ndcg_at_10": 0.5,
            "bytes_per_vector": 64.0,
            "bytes_per_vector_measured": 64.0,
        },
    }
    assert (
        rule_based(RunOutcome(True, "evaluate", record=rec), results).verdict == "accept_unverified"
    )
    bad = json.loads(json.dumps(rec))
    bad["metrics"]["bytes_per_vector_measured"] = 65.0
    assert rule_based(RunOutcome(True, "evaluate", record=bad), results).verdict == "reject"
    nondet = json.loads(json.dumps(rec))
    nondet["integrity"]["deterministic"] = False
    assert rule_based(RunOutcome(True, "evaluate", record=nondet), results).verdict == "reject"
    assert (
        rule_based(
            RunOutcome(False, "tampered", eval_problems_after=["modified: x.py"]), results
        ).verdict
        == "tampered"
    )
    assert (
        rule_based(RunOutcome(False, "static_check", static_violations=["bad"]), results).verdict
        == "reject"
    )


def _lab(tmp_path: Path, llm: FakeLLM) -> Lab:
    paths = LabPaths(
        lab_dir=tmp_path / "lab",
        results_dir=tmp_path / "results",
        reports_dir=tmp_path / "reports",
        docs_dir=tmp_path / "docs",
    )
    cfg = LabConfig(sandbox_timeout_s=300, paths=paths, librarian_max_per_query=5)
    return Lab(
        cfg,
        llm,
        lambda: toy_prepared(n=200),
        fetchers=[fake_papers],
        embedder=lib.hash_embedder(),
        ideas_md="- try a lower alpha",
    )


def test_full_cycle_with_fake_llm_and_baselines(tmp_path: Path) -> None:
    # baselines first, so the critic has something to compare against
    cfg_path = tmp_path / "toy.yaml"
    cfg_path.write_text(
        "name: toy\ndataset: toy\nmodel: {name: none, revision: null}\nk: 10\nseeds: [0]\nwarmup: 1\n"
        "quantizers: [{name: float32}, {name: binary}, {name: ternary, params: {alpha: 0.7}}]\n",
        encoding="utf-8",
    )
    run_bench(cfg_path, results_dir=tmp_path / "results", data=toy_prepared(n=200))

    llm = FakeLLM([PROPOSAL, CODE, REVIEW])
    lab = _lab(tmp_path, llm)
    status = lab.run_cycle()
    assert status["experiment_id"] == "exp_0001"
    assert status["verdict"] in ("accept_unverified", "suspicious"), status

    exp = tmp_path / "lab" / "experiments" / "exp_0001"
    for name in (
        "spec.md",
        "spec.json",
        "quantizer.py",
        "test_quantizer.py",
        "result.json",
        "critique.md",
        "logged.json",
    ):
        assert (exp / name).exists(), name
    lab_results = [p for p in (tmp_path / "results").glob("*.json") if "half_ternary" in p.name]
    assert len(lab_results) == 1
    rec = json.loads(lab_results[0].read_text())
    assert (
        rec["source"] == "lab"
        and rec["experiment_id"] == "exp_0001"
        and rec["verification"] == "unverified"
    )

    reports = list((tmp_path / "reports").glob("*.md"))
    assert len(reports) == 1 and "half_ternary" in reports[0].read_text(encoding="utf-8")
    assert (tmp_path / "docs" / "dashboard" / "index.html").exists()
    assert len(llm.calls) == 3  # proposer, coder, critic review — no hidden model calls

    # the proposer saw the ideas, the notes and the baseline table
    prompt = llm.calls[0][1].content
    assert (
        "lower alpha" in prompt
        and "Ternary quantization of embeddings" in prompt
        and "float32" in prompt
    )

    # manual approval is the only way to a label
    approve(tmp_path / "results", rec["run_id"], "reproduced")
    assert load_verification(tmp_path / "results")[rec["run_id"]] == "reproduced"
    st = lab_status(tmp_path / "lab")
    assert st["state"]["stage"] == "done" and st["experiments"]["exp_0001"] == status["verdict"]


def test_resume_after_crash_in_coder_stage(tmp_path: Path) -> None:
    llm = FakeLLM([PROPOSAL])  # enough for the proposer only
    lab = _lab(tmp_path, llm)
    # run librarian + proposer, then simulate a crash before the coder finished
    state = LabState("exp_0001", "librarian", "t", "t")
    state.save(lab.paths.state_file)
    exp = lab.exp_dir("exp_0001")
    lab.stage_librarian(exp)
    spec = lab.stage_proposer(exp)
    state.stage = "coder"
    state.save(lab.paths.state_file)
    assert spec.quantizer_name == "half_ternary" and not (exp / "quantizer.py").exists()

    # restart with a fresh Lab object: it must resume at "coder" and not re-propose
    llm2 = FakeLLM([CODE, REVIEW])
    lab2 = _lab(tmp_path, llm2)
    status = lab2.run_cycle()
    assert status["experiment_id"] == "exp_0001" and status["verdict"] in (
        "accept_unverified",
        "suspicious",
    )
    assert len(llm2.calls) == 2 and "Coder" in llm2.calls[0][0].content


def test_llm_failure_is_recorded_not_raised(tmp_path: Path) -> None:
    lab = _lab(tmp_path, FakeLLM(["this is not json at all"]))
    status = lab.run_cycle()
    assert status["verdict"] == "reject"
    crit = (tmp_path / "lab" / "experiments" / "exp_0001" / "critique.md").read_text(
        encoding="utf-8"
    )
    assert "LLM stage failed" in crit
    assert LabState.load(lab.paths.state_file).stage == "done"  # type: ignore[union-attr]
