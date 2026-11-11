"""Entry point of the sandbox subprocess: `python -I -m trivec.lab.sandbox_entry <job.json>`.

Trusted code. It installs the network guard, loads the agent's quantizer module from a
file path, runs integrity checks, runs the FROZEN evaluator, and writes one result JSON.
The parent (runner.py) checks the evaluator's hashes before and after.
"""

import hashlib
import importlib.util
import json
import sys
import traceback
import types
from pathlib import Path
from typing import Any

from trivec.lab import netguard

GROWTH_LIMIT_BYTES_PER_VECTOR = 1.0


def load_agent_module(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("agent_quantizer", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _state_size(quantizer: Any, module: Any) -> int:
    """Bytes held by the instance plus anything stashed in the module's global variables."""
    from trivec.eval.memory import deep_size

    seen: set[int] = set()
    size = deep_size(quantizer, seen)
    for name, value in vars(module).items():
        if not name.startswith("__") and not isinstance(value, types.ModuleType):
            size += deep_size(value, seen)
    return size


def integrity_checks(module: Any, params: dict[str, Any], corpus: Any, seed: int) -> dict[str, Any]:
    """Catch quantizers whose numbers would be meaningless, however good they look.

    deterministic      : same seed, same codes.
    decode_follows_codes: reordering the codes must reorder the output. A decode that ignores
                         its input and returns stored vectors fails this.
    state_growth       : hidden state must not grow with the corpus. A quantizer that memorises
                         vectors outside its codes grows; a codebook of fixed size does not.
    """
    import numpy as np

    n = int(corpus.shape[0])
    sample = corpus[: min(256, n)]

    first = module.build(**params)
    first.fit(corpus, seed=seed)
    codes = first.encode(sample)
    second = module.build(**params)
    second.fit(corpus, seed=seed)
    deterministic = bool(np.array_equal(codes, second.encode(sample)))

    perm = np.random.default_rng(0).permutation(codes.shape[0])
    decode_ok = bool(np.array_equal(first.decode(codes[perm]), first.decode(codes)[perm]))

    small_n = max(n // 4, 1)
    small = load_agent_module(Path(module.__file__))
    q_small = small.build(**params)
    q_small.fit(corpus[:small_n], seed=seed)
    q_small.encode(corpus[:small_n])
    small_size = _state_size(q_small, small)

    large = load_agent_module(Path(module.__file__))
    q_large = large.build(**params)
    q_large.fit(corpus, seed=seed)
    q_large.encode(corpus)
    large_size = _state_size(q_large, large)

    growth = (large_size - small_size) / max(n - small_n, 1)
    return {
        "deterministic": deterministic,
        "decode_follows_codes": decode_ok,
        "state_growth_bytes_per_vector": round(growth, 3),
        "state_growth_ok": growth <= GROWTH_LIMIT_BYTES_PER_VECTOR,
    }


def main(job_path: str) -> int:
    netguard.install()
    job = json.loads(Path(job_path).read_text(encoding="utf-8"))
    out_path = Path(job["out_json"])
    record: dict[str, Any]
    try:
        from trivec.data.prepare import load_prepared
        from trivec.eval.evaluate import RunSpec, evaluate

        data = load_prepared(Path(job["data_dir"]))
        source = Path(job["quantizer_path"])
        module = load_agent_module(source)
        if not hasattr(module, "build"):
            raise AttributeError("agent module has no build(**params) function")
        params = dict(job.get("params", {}))

        integrity = integrity_checks(module, params, data.corpus, int(job["seed"]))
        spec = RunSpec(
            quantizer_name=str(job["quantizer_name"]),
            quantizer_params=params,
            seed=int(job["seed"]),
            k=int(job.get("k", 10)),
            search_mode=str(job.get("search_mode", "asymmetric_flat")),
            warmup=int(job.get("warmup", 20)),
            config=dict(job.get("config", {})),
        )
        record = evaluate(spec, data, module.build(**params))
        record["integrity"] = integrity
        # git does not cover agent code at run time, so the result pins it by hash (hard rule 2)
        record["quantizer_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    except Exception as exc:
        record = {
            "status": "failed",
            "quantizer": job.get("quantizer_name"),
            "quantizer_params": job.get("params", {}),
            "search_mode": job.get("search_mode", "asymmetric_flat"),
            "seed": job.get("seed"),
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            },
        }
    record["source"] = "lab"
    record["experiment_id"] = job.get("experiment_id")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
