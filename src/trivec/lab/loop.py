"""The Lab loop: librarian -> proposer -> coder -> runner -> critic -> logger.

Every stage writes its output to disk atomically; `state.json` records the stage. On
start-up we re-run the stage we were in, so a crash anywhere is safe (docs/design.md 7.2).
"""

import datetime as dt
import json
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

from trivec.data.prepare import PreparedData
from trivec.eval.evaluate import write_result
from trivec.lab import library as lib
from trivec.lab.agents import Spec, code, propose
from trivec.lab.config import LabConfig
from trivec.lab.critic import Critique, critique, rule_based
from trivec.lab.llm import LLM, LLMError
from trivec.lab.runner import RunOutcome, run_experiment
from trivec.lab.state import (
    LabState,
    atomic_write_json,
    atomic_write_text,
    next_experiment_id,
    now_iso,
    read_json,
)
from trivec.quantizers import REGISTRY
from trivec.report import aggregate, load_results, markdown_table, run_report

DataProvider = Callable[[], PreparedData]


def results_summary(results_dir: Path, dataset: str) -> str:
    groups = aggregate([r for r in load_results(results_dir) if r.get("status") == "ok"])
    parts = []
    for key, rows in groups.items():
        if key[0] == dataset:
            parts.append(f"### {key[0]} (k={key[2]})\n" + markdown_table(key, rows))
    return "\n".join(parts) or "(no results yet)"


def existing_quantizer_names(experiments_dir: Path) -> list[str]:
    names = set(REGISTRY) | {"pq"}
    for spec_path in experiments_dir.glob("*/spec.json"):
        names.add(str(read_json(spec_path, {}).get("quantizer_name", "")))
    return sorted(n for n in names if n)


def past_critiques(experiments_dir: Path) -> list[str]:
    return [p.read_text(encoding="utf-8") for p in sorted(experiments_dir.glob("*/critique.md"))]


class Lab:
    def __init__(
        self,
        cfg: LabConfig,
        llm: LLM,
        data_provider: DataProvider,
        *,
        fetchers: list[lib.Fetcher] | None = None,
        embedder: lib.Embedder | None = None,
        ideas_md: str | None = None,
    ) -> None:
        self.cfg = cfg
        self.llm = llm
        self.data_provider = data_provider
        self.paths = cfg.paths
        self.library = lib.Library(self.paths.library_dir)
        self.fetchers = (
            fetchers if fetchers is not None else [lib.fetch_arxiv, lib.fetch_semantic_scholar]
        )
        self.embedder = embedder or (
            lib.hash_embedder()
            if cfg.librarian_embedder == "hash"
            else lib.model_embedder(cfg.librarian_embedder)
        )
        self.ideas_md = (
            ideas_md if ideas_md is not None else self._read(self.paths.docs_dir / "ideas.md")
        )
        self._data: PreparedData | None = None

    @staticmethod
    def _read(path: Path) -> str:
        return path.read_text(encoding="utf-8") if path.exists() else ""

    def data(self) -> PreparedData:
        if self._data is None:
            self._data = self.data_provider()
        return self._data

    def exp_dir(self, experiment_id: str) -> Path:
        d = self.paths.experiments_dir / experiment_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    # --- stages (each idempotent: skipped when its output exists) ---------------------------

    def stage_librarian(self, exp: Path) -> None:
        marker = exp / "librarian.json"
        if marker.exists():
            return
        added = 0
        if self.cfg.librarian_enabled and self.fetchers:
            added = self.library.collect(
                self.cfg.librarian_queries, self.cfg.librarian_max_per_query, self.fetchers
            )
        atomic_write_json(
            marker,
            {"notes_added": added, "notes_total": len(self.library.notes()), "at": now_iso()},
        )

    def stage_proposer(self, exp: Path) -> Spec:
        spec_path = exp / "spec.json"
        if spec_path.exists():
            return Spec(**read_json(spec_path))
        dataset = self.data().dataset_name
        query = "ternary binary quantization embeddings recall memory"
        notes = self.library.search(query, self.embedder, k=5)
        existing = existing_quantizer_names(self.paths.experiments_dir)
        spec, raw = propose(
            self.llm,
            ideas_md=self.ideas_md,
            results_summary=results_summary(self.paths.results_dir, dataset),
            notes=notes,
            past_critiques=past_critiques(self.paths.experiments_dir),
            existing_names=existing,
        )
        atomic_write_text(exp / "proposer_reply.txt", raw)
        atomic_write_text(exp / "spec.md", spec.to_markdown())
        atomic_write_json(spec_path, spec.__dict__)
        return spec

    def stage_coder(self, exp: Path, spec: Spec) -> None:
        if (exp / "quantizer.py").exists():
            return
        quantizer_py, test_py, transcript = code(self.llm, spec)
        atomic_write_text(exp / "coder_replies.txt", "\n\n=====\n\n".join(transcript))
        atomic_write_text(exp / "test_quantizer.py", test_py)
        atomic_write_text(
            exp / "quantizer.py", quantizer_py
        )  # last: its presence marks the stage done

    def stage_runner(self, exp: Path, spec: Spec, experiment_id: str) -> RunOutcome:
        out_path = exp / "run_outcome.json"
        if out_path.exists():
            d = read_json(out_path)
            return RunOutcome(**d)
        outcome = run_experiment(
            exp,
            self.data(),
            quantizer_name=spec.quantizer_name,
            params=spec.params,
            seed=self.cfg.seed,
            search_mode=self.cfg.search_mode,
            timeout_s=self.cfg.sandbox_timeout_s,
            config={
                "name": f"lab_{experiment_id}",
                "spec": spec.__dict__,
                "bench_config": str(self.cfg.bench_config),
            },
            experiment_id=experiment_id,
        )
        if outcome.record is not None:
            # copy into results/ with the lab marker; failed runs are kept too (hard rule 9)
            rid = (
                outcome.record.get("run_id")
                or f"{experiment_id}_failed_{dt.datetime.now(dt.UTC):%Y%m%dT%H%M%S}"
            )
            outcome.record["run_id"] = rid
            write_result(outcome.record, self.paths.results_dir)
        atomic_write_json(out_path, outcome.__dict__)
        return outcome

    def stage_critic(self, exp: Path, outcome: RunOutcome) -> Critique:
        path = exp / "critique.md"
        json_path = exp / "critique.json"
        if json_path.exists():
            d = read_json(json_path)
            return Critique(**d)
        c = critique(
            outcome,
            self.paths.results_dir,
            llm=self.llm,
            spec_md=self._read(exp / "spec.md"),
            quantizer_py=self._read(exp / "quantizer.py"),
        )
        atomic_write_text(path, c.to_markdown())
        atomic_write_json(json_path, c.__dict__)
        return c

    def stage_logger(
        self,
        exp: Path,
        experiment_id: str,
        spec: Spec | None,
        outcome: RunOutcome | None,
        crit: Critique,
    ) -> None:
        marker = exp / "logged.json"
        if marker.exists():
            return
        self.paths.reports_dir.mkdir(parents=True, exist_ok=True)
        report = self.paths.reports_dir / f"{dt.date.today():%Y-%m-%d}.md"
        header = (
            ""
            if report.exists()
            else f"# Lab report {dt.date.today():%Y-%m-%d}\n\nPlain-English log of what the lab tried today. Nothing here is verified unless the dashboard says so.\n"
        )
        rec = outcome.record if outcome else None
        m = (rec or {}).get("metrics") or {}
        k = (rec or {}).get("k", 10)
        what = (
            f"**{spec.title}** (`{spec.quantizer_name}`): {spec.hypothesis}"
            if spec
            else "(no proposal)"
        )
        if rec and rec.get("status") == "ok":
            result = (
                f"recall@{k} {m.get(f'recall_at_{k}', 0):.4f}, nDCG@{k} {m.get(f'ndcg_at_{k}', 0):.4f}, "
                f"{m.get('bytes_per_vector', 0):.0f} bytes/vector, p50 {m.get('p50_latency_ms', 0):.2f} ms "
                f"on {rec['dataset']['name']} (one seed)"
            )
        else:
            result = f"did not produce a result ({outcome.message if outcome else 'no run'})"
        entry = (
            f"{header}\n## {experiment_id}\n\n- What we tried: {what}\n- Result: {result}\n"
            f"- Critic verdict: `{crit.verdict}` — {'; '.join(crit.reasons[:3])}\n"
            f"- Status on dashboard: unverified\n"
        )
        with report.open("a", encoding="utf-8") as f:
            f.write(entry)
        run_report(self.paths.results_dir, self.paths.docs_dir, dashboard=True)
        atomic_write_json(marker, {"at": now_iso(), "report": str(report)})

    # --- the cycle -------------------------------------------------------------------------

    def run_cycle(self) -> dict[str, Any]:
        state = LabState.load(self.paths.state_file)
        if state is None or state.stage == "done":
            experiment_id = next_experiment_id(self.paths.experiments_dir)
            state = LabState(experiment_id, "librarian", now_iso(), now_iso())
            state.save(self.paths.state_file)
        exp = self.exp_dir(state.experiment_id)
        spec: Spec | None = (
            Spec(**read_json(exp / "spec.json")) if (exp / "spec.json").exists() else None
        )
        outcome: RunOutcome | None = (
            RunOutcome(**read_json(exp / "run_outcome.json"))
            if (exp / "run_outcome.json").exists()
            else None
        )
        crit: Critique | None = None

        try:
            while state.stage != "done":
                if state.stage == "librarian":
                    self.stage_librarian(exp)
                elif state.stage == "proposer":
                    spec = self.stage_proposer(exp)
                elif state.stage == "coder":
                    assert spec is not None
                    self.stage_coder(exp, spec)
                elif state.stage == "runner":
                    assert spec is not None
                    outcome = self.stage_runner(exp, spec, state.experiment_id)
                elif state.stage == "critic":
                    assert outcome is not None
                    crit = self.stage_critic(exp, outcome)
                elif state.stage == "logger":
                    if crit is None:
                        crit = Critique(**read_json(exp / "critique.json"))
                    self.stage_logger(exp, state.experiment_id, spec, outcome, crit)
                state.advance(self.paths.state_file)
        except LLMError as exc:
            # a model failure ends this experiment as a recorded rejection, never a crash loop
            crit = Critique("reject", [f"LLM stage failed: {exc}"])
            atomic_write_text(exp / "critique.md", crit.to_markdown())
            atomic_write_json(exp / "critique.json", crit.__dict__)
            self.stage_logger(exp, state.experiment_id, spec, outcome, crit)
            state.stage = "done"
            state.save(self.paths.state_file)

        status = {"experiment_id": state.experiment_id, "verdict": crit.verdict if crit else None}
        atomic_write_json(exp / "status.json", status)
        return status


def lab_status(paths_lab_dir: Path) -> dict[str, Any]:
    state = read_json(paths_lab_dir / "state.json", {})
    exps = (
        sorted(p.name for p in (paths_lab_dir / "experiments").glob("exp_*"))
        if (paths_lab_dir / "experiments").exists()
        else []
    )
    verdicts = {}
    for e in exps:
        c = read_json(paths_lab_dir / "experiments" / e / "critique.json", {})
        verdicts[e] = c.get("verdict")
    return {"state": state, "experiments": verdicts}


def reproduce(
    cfg: LabConfig,
    experiment_id: str,
    primary: PreparedData,
    holdouts: list[PreparedData],
    *,
    seeds: tuple[int, ...] = (1, 2, 3),
) -> dict[str, Any]:
    """Re-run one experiment on new seeds and on datasets the lab never tuned against.

    Writes reproduction.json. `passed` means every run was clean. It does NOT mean the idea
    wins: that is `beats_baselines_everywhere`, and the decision stays with a human.
    """
    exp = cfg.paths.experiments_dir / experiment_id
    spec = Spec(**read_json(exp / "spec.json"))
    targets = [(primary, s) for s in seeds] + [(d, seeds[0]) for d in holdouts]
    runs: list[dict[str, Any]] = []
    for data, seed in targets:
        outcome = run_experiment(
            exp,
            data,
            quantizer_name=spec.quantizer_name,
            params=spec.params,
            seed=seed,
            search_mode=cfg.search_mode,
            timeout_s=cfg.sandbox_timeout_s,
            config={"name": f"lab_{experiment_id}_reproduce", "spec": spec.__dict__},
            experiment_id=experiment_id,
        )
        crit = rule_based(outcome, cfg.paths.results_dir)
        if outcome.record is not None:
            outcome.record.setdefault(
                "run_id", f"{experiment_id}_failed_{dt.datetime.now(dt.UTC):%Y%m%dT%H%M%S}"
            )
            write_result(outcome.record, cfg.paths.results_dir)
        runs.append(
            {
                "dataset": data.dataset_name,
                "seed": seed,
                "verdict": crit.verdict,
                "reasons": crit.reasons[:3],
                "recall_gain_vs_best": crit.comparison.get("recall_gain_vs_best"),
            }
        )
    gains = [r["recall_gain_vs_best"] for r in runs]
    out: dict[str, Any] = {
        "experiment_id": experiment_id,
        "at": now_iso(),
        "runs": runs,
        "passed": all(r["verdict"] in ("accept_unverified", "suspicious") for r in runs),
        "beats_baselines_everywhere": all(g is not None and g > 0 for g in gains),
    }
    atomic_write_json(exp / "reproduction.json", out)
    return out


def approve(
    results_dir: Path, run_id: str, label: str = "verified", *, lab_dir: Path = Path("lab")
) -> dict[str, str]:
    """Your manual gate. Writes results/verification.json; the lab itself never calls this.

    A lab result cannot become "verified" until its reproduction run has passed.
    """
    if label not in ("unverified", "reproduced", "verified"):
        raise ValueError("label must be unverified | reproduced | verified")
    result_path = results_dir / f"{run_id}.json"
    if not result_path.exists():
        raise FileNotFoundError(f"no result {run_id}")
    record = json.loads(result_path.read_text(encoding="utf-8"))
    if label == "verified" and record.get("source") == "lab":
        repro = read_json(
            lab_dir / "experiments" / str(record.get("experiment_id")) / "reproduction.json", {}
        )
        if not repro.get("passed"):
            raise ValueError(
                f"{run_id} has no passing reproduction run: run `trivec lab reproduce` first"
            )
    path = results_dir / "verification.json"
    data: dict[str, str] = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    data[run_id] = label
    atomic_write_json(path, data)
    return data


def clear_lab_state(lab_dir: Path) -> None:
    if (lab_dir / "state.json").exists():
        (lab_dir / "state.json").unlink()
    shutil.rmtree(lab_dir / "experiments", ignore_errors=True)
