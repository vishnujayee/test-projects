"""Command-line entry point: `trivec bench | report | lab`."""

import os
from pathlib import Path

import typer

from trivec import __version__

app = typer.Typer(help="TriVec Lab command line.", no_args_is_help=True)


def _one_blas_thread() -> None:
    """Must run before numpy loads. One thread makes timings comparable and saves a lot of RAM:
    OpenBLAS allocates a buffer per thread, which fails on a small machine."""
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(name, "1")


@app.callback()
def main() -> None:
    """TriVec Lab: benchmark and research lab for extreme vector compression."""


@app.command()
def info() -> None:
    """Print the installed version."""
    typer.echo(f"trivec {__version__}")


@app.command()
def bench(
    config: Path = typer.Option(..., "--config", "-c", exists=True, help="configs/<name>.yaml"),
    results_dir: Path = typer.Option(Path("results"), help="where result JSON files go"),
    cache_root: Path = typer.Option(Path("data") / "cache", help="embedding cache root"),
) -> None:
    """Run every quantizer x seed x mode in a config and write one JSON per run."""
    _one_blas_thread()
    from trivec.bench import run_bench

    written = run_bench(config, results_dir=results_dir, cache_root=cache_root)
    typer.echo(f"wrote {len(written)} result files to {results_dir}")


@app.command()
def prepare(
    config: Path = typer.Option(..., "--config", "-c", exists=True, help="configs/<name>.yaml"),
    cache_root: Path = typer.Option(Path("data") / "cache", help="embedding cache root"),
) -> None:
    """Download, embed and cache a config's dataset. Run it alone: embedding needs RAM."""
    _one_blas_thread()
    from trivec.bench import load_config
    from trivec.data.prepare import prepare as prepare_data

    cfg = load_config(config)
    data = prepare_data(
        cfg.dataset,
        cfg.model_name,
        model_revision=cfg.model_revision,
        k=cfg.k,
        cache_root=cache_root,
    )
    typer.echo(
        f"{cfg.dataset}: {data.corpus.shape[0]} docs, {data.queries.shape[0]} queries, "
        f"dim={data.dim}, dataset_rev={data.dataset_revision[:12]}, "
        f"model_rev={data.model_revision[:12]}"
    )


@app.command("break")
def break_cmd(
    configs: list[Path] = typer.Option(
        None, "--config", "-c", help="default: the three main configs"
    ),
    out: Path = typer.Option(Path("results") / "checks" / "break_tests.json"),
) -> None:
    """Try to break the benchmark: leaks, free hits, axis dependence, tie order, scale."""
    _one_blas_thread()
    from trivec.break_tests import run_break_tests

    default = [Path(f"configs/{d}_minilm.yaml") for d in ("scifact", "nfcorpus", "arguana")]
    run_break_tests(list(configs) if configs else default, out)
    typer.echo(f"wrote {out}")


@app.command()
def report(
    results_dir: Path = typer.Option(Path("results")),
    docs_dir: Path = typer.Option(Path("docs")),
    dashboard: bool = typer.Option(True, help="also write docs/dashboard/index.html"),
) -> None:
    """Regenerate docs/results.md, charts and the dashboard from results/ only."""
    _one_blas_thread()
    from trivec.report import run_report

    info_ = run_report(results_dir, docs_dir, dashboard=dashboard)
    n_charts = len(info_["charts"])
    typer.echo(f"{info_['n_records']} records, {info_['n_groups']} groups, {n_charts} charts")


lab_app = typer.Typer(help="The agent loop (Phase 2).", no_args_is_help=True)
app.add_typer(lab_app, name="lab")


@lab_app.command("run")
def lab_run(
    config: Path = typer.Option(Path("configs/lab.yaml"), "--config", "-c", exists=True),
    cycles: int = typer.Option(1, help="how many experiments to run"),
    backend: str | None = typer.Option(
        None, help="override llm backend: ollama|anthropic|openai_compatible"
    ),
) -> None:
    """Run the lab loop: librarian -> proposer -> coder -> sandboxed runner -> critic -> logger."""
    _one_blas_thread()
    import dataclasses

    from trivec.bench import load_config
    from trivec.data.prepare import prepare as prepare_data
    from trivec.lab.config import load_lab_config
    from trivec.lab.llm import get_llm
    from trivec.lab.loop import Lab

    cfg = load_lab_config(config)
    if backend:
        cfg = dataclasses.replace(cfg, llm=dataclasses.replace(cfg.llm, backend=backend))
    bench = load_config(cfg.bench_config)

    def provider():  # type: ignore[no-untyped-def]
        return prepare_data(
            bench.dataset, bench.model_name, model_revision=bench.model_revision, k=bench.k
        )

    lab = Lab(cfg, get_llm(cfg.llm), provider)
    for _ in range(cycles):
        status = lab.run_cycle()
        typer.echo(f"{status['experiment_id']}: verdict={status['verdict']}")


@lab_app.command("status")
def lab_status_cmd(lab_dir: Path = typer.Option(Path("lab"))) -> None:
    """Show the current stage and every experiment's verdict."""
    import json

    from trivec.lab.loop import lab_status

    typer.echo(json.dumps(lab_status(lab_dir), indent=2))


@lab_app.command("approve")
def lab_approve(
    run_id: str = typer.Argument(..., help="run id (file name without .json)"),
    label: str = typer.Option("verified", help="unverified | reproduced | verified"),
    results_dir: Path = typer.Option(Path("results")),
    lab_dir: Path = typer.Option(Path("lab")),
) -> None:
    """Your manual gate: label a run. The lab never does this by itself."""
    from trivec.lab.loop import approve

    approve(results_dir, run_id, label, lab_dir=lab_dir)
    typer.echo(f"{run_id} -> {label}. Run `trivec report` to refresh the dashboard.")


@lab_app.command("reproduce")
def lab_reproduce(
    experiment_id: str = typer.Argument(..., help="for example exp_0001"),
    config: Path = typer.Option(Path("configs/lab.yaml"), "--config", "-c", exists=True),
) -> None:
    """Re-run an experiment on new seeds and held-out datasets. Required before `approve`."""
    _one_blas_thread()
    from trivec.bench import load_config
    from trivec.data.prepare import prepare as prepare_data
    from trivec.lab.config import load_lab_config
    from trivec.lab.loop import reproduce

    cfg = load_lab_config(config)

    def provider(path: Path):  # type: ignore[no-untyped-def]
        bench_cfg = load_config(path)
        return prepare_data(
            bench_cfg.dataset,
            bench_cfg.model_name,
            model_revision=bench_cfg.model_revision,
            k=bench_cfg.k,
        )

    out = reproduce(
        cfg, experiment_id, provider(cfg.bench_config), [provider(p) for p in cfg.holdout_configs]
    )
    wins = out["beats_baselines_everywhere"]
    typer.echo(f"{experiment_id}: passed={out['passed']}, wins everywhere={wins}")


@lab_app.command("librarian")
def lab_librarian(
    config: Path = typer.Option(Path("configs/lab.yaml"), "--config", "-c", exists=True),
) -> None:
    """Fetch papers on our topic and write notes (needs network)."""
    from trivec.lab import library as lib
    from trivec.lab.config import load_lab_config

    cfg = load_lab_config(config)
    library = lib.Library(cfg.paths.library_dir)
    added = library.collect(
        cfg.librarian_queries,
        cfg.librarian_max_per_query,
        [lib.fetch_arxiv, lib.fetch_semantic_scholar],
    )
    typer.echo(f"added {added} notes; total {len(library.notes())}")


if __name__ == "__main__":
    app()
