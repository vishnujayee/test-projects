"""Freeze check for the evaluator (docs/design.md section 6.4).

FROZEN.sha256 lists "<sha256>  <filename>" for every .py file in this package.
`check()` recomputes and compares. CI runs it as a test; the Lab runner calls it before
and after every agent experiment. Only an explicit "unfreeze eval" updates the file.
"""

import hashlib
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
HASH_FILE = EVAL_DIR / "FROZEN.sha256"


def current_hashes(eval_dir: Path = EVAL_DIR) -> dict[str, str]:
    out: dict[str, str] = {}
    for path in sorted(eval_dir.glob("*.py")):
        data = path.read_bytes().replace(b"\r\n", b"\n")  # line endings must not matter
        out[path.name] = hashlib.sha256(data).hexdigest()
    return out


def read_hashes(hash_file: Path = HASH_FILE) -> dict[str, str]:
    if not hash_file.exists():
        return {}
    out: dict[str, str] = {}
    for line in hash_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            digest, name = line.split(None, 1)
            out[name.strip()] = digest
    return out


def write_hashes(eval_dir: Path = EVAL_DIR, hash_file: Path = HASH_FILE) -> None:
    lines = ["# sha256 of every .py in src/trivec/eval/. Do not edit without 'unfreeze eval'."]
    lines += [f"{d}  {n}" for n, d in current_hashes(eval_dir).items()]
    hash_file.write_text("\n".join(lines) + "\n", encoding="utf-8")


def check(eval_dir: Path = EVAL_DIR, hash_file: Path = HASH_FILE) -> list[str]:
    """Return a list of problems; empty means the evaluator is intact."""
    recorded = read_hashes(hash_file)
    if not recorded:
        return ["FROZEN.sha256 is missing or empty"]
    now = current_hashes(eval_dir)
    problems = [f"modified: {n}" for n, d in now.items() if recorded.get(n) not in (None, d)]
    problems += [f"added: {n}" for n in now if n not in recorded]
    problems += [f"deleted: {n}" for n in recorded if n not in now]
    return problems


def is_frozen_intact() -> bool:
    return not check()
