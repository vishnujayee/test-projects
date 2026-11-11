"""Proposer and Coder: prompt building and reply parsing. No direct model calls here;
everything goes through the LLM plug (`llm.complete`)."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from trivec.lab.llm import LLM, LLMError, Message, extract_json
from trivec.lab.static_check import check_source, has_required_contract

PKG = Path(__file__).resolve().parents[1]
BASE_SRC = (PKG / "quantizers" / "base.py").read_text(encoding="utf-8")
EXAMPLE_SRC = (PKG / "quantizers" / "ternary.py").read_text(encoding="utf-8")
BITS_SRC = (PKG / "quantizers" / "bits.py").read_text(encoding="utf-8")


@dataclass
class Spec:
    title: str
    hypothesis: str
    quantizer_name: str
    method: str
    params: dict[str, Any] = field(default_factory=dict)
    expected_bytes_per_vector: str = ""
    risks: str = ""

    def to_markdown(self) -> str:
        return (
            f"# {self.title}\n\n**Quantizer name:** `{self.quantizer_name}`\n\n"
            f"**Hypothesis.** {self.hypothesis}\n\n**Method.** {self.method}\n\n"
            f"**Params.** `{json.dumps(self.params)}`\n\n"
            f"**Expected bytes/vector.** {self.expected_bytes_per_vector}\n\n**Risks.** {self.risks}\n"
        )


PROPOSER_SYSTEM = """You are the Proposer in TriVec Lab, a research lab for extreme vector compression
(ternary / binary / low-bit embeddings for retrieval). Propose exactly ONE small experiment.
Rules:
- It must be a NEW quantizer implementable in pure numpy against the Quantizer interface below.
- Prefer ideas from the idea list and paper notes. Build on past results; do not repeat a name that exists.
- Keep it simple enough to implement in under 120 lines.
- Reply with ONE JSON object only, keys: title, hypothesis, quantizer_name (snake_case, unique),
  method (precise description of fit/encode/decode, 3-8 sentences), params (object of numbers/strings),
  expected_bytes_per_vector (formula in d), risks (one or two sentences)."""


def build_proposer_messages(
    *,
    ideas_md: str,
    results_summary: str,
    notes: list[tuple[str, str]],
    past_critiques: list[str],
    existing_names: list[str],
) -> list[Message]:
    notes_txt = (
        "\n\n".join(f"### note {nid}\n{text[:1500]}" for nid, text in notes) or "(no notes yet)"
    )
    crit_txt = "\n\n".join(c[:1200] for c in past_critiques[-3:]) or "(none yet)"
    user = (
        f"## Quantizer interface\n```python\n{BASE_SRC}\n```\n\n"
        f"## Existing quantizer names (do not reuse)\n{', '.join(existing_names)}\n\n"
        f"## Ideas list (docs/ideas.md)\n{ideas_md[:4000]}\n\n"
        f"## Current results (mean over seeds)\n{results_summary[:4000]}\n\n"
        f"## Relevant paper notes\n{notes_txt}\n\n"
        f"## Recent critic reports\n{crit_txt}\n\n"
        "Propose one experiment now. JSON only."
    )
    return [Message("system", PROPOSER_SYSTEM), Message("user", user)]


def parse_spec(reply: str, existing_names: list[str]) -> Spec:
    obj = extract_json(reply)
    required = ("title", "hypothesis", "quantizer_name", "method")
    missing = [k for k in required if not obj.get(k)]
    if missing:
        raise LLMError(f"proposal missing keys: {missing}")
    name = str(obj["quantizer_name"]).strip().lower().replace("-", "_").replace(" ", "_")
    if name in existing_names:
        raise LLMError(f"proposal reuses existing quantizer name {name!r}")
    params = obj.get("params") or {}
    if not isinstance(params, dict):
        raise LLMError("params must be an object")
    return Spec(
        title=str(obj["title"]),
        hypothesis=str(obj["hypothesis"]),
        quantizer_name=name,
        method=str(obj["method"]),
        params=params,
        expected_bytes_per_vector=str(obj.get("expected_bytes_per_vector", "")),
        risks=str(obj.get("risks", "")),
    )


def propose(llm: LLM, **ctx: Any) -> tuple[Spec, str]:
    existing = list(ctx["existing_names"])
    reply = llm.complete(build_proposer_messages(**ctx), json_mode=True)
    return parse_spec(reply, existing), reply


CODER_SYSTEM = """You are the Coder in TriVec Lab. Implement the given experiment spec as ONE Python module
plus ONE pytest file. Reply with ONE JSON object only: {"quantizer_py": "<source>", "test_py": "<source>"}.

Hard rules for quantizer_py (it runs in a sandbox and is rejected otherwise):
- Allowed imports ONLY: numpy, math, typing, dataclasses, collections.abc,
  trivec.quantizers.base, trivec.quantizers.bits. Nothing else. No file or network access.
- Define a class with a class attribute `name` equal to the quantizer_name from the spec, and the
  methods fit(self, corpus, *, seed), encode(self, x), decode(self, codes), bytes_per_vector(self, dim).
- Define a module-level function `def build(**params)` that returns an instance.
- fit sees ONLY the corpus. encode/decode must be deterministic. decode returns float32 [n, d].
- codes.nbytes / n must equal bytes_per_vector(dim) exactly (include per-vector scales you store).
- No global statements, no eval/exec/open.
Rules for test_py: use only pytest and numpy; build a random float32 corpus with np.random.default_rng(0),
L2-normalise it, and test: output shape, dtype, determinism across two builds with the same seed,
decode shape/dtype/finite, and bytes_per_vector == codes.nbytes / n."""


def build_coder_messages(spec: Spec, feedback: str | None = None) -> list[Message]:
    user = (
        f"## Spec\n{spec.to_markdown()}\n\n"
        f"## Interface (trivec/quantizers/base.py)\n```python\n{BASE_SRC}\n```\n\n"
        f"## Helpers you may import (trivec/quantizers/bits.py)\n```python\n{BITS_SRC}\n```\n\n"
        f"## Example implementation (trivec/quantizers/ternary.py)\n```python\n{EXAMPLE_SRC}\n```\n\n"
        f'The class attribute `name` must be exactly "{spec.quantizer_name}". '
        f"build(**params) will be called with params={json.dumps(spec.params)}.\nJSON only."
    )
    msgs = [Message("system", CODER_SYSTEM), Message("user", user)]
    if feedback:
        msgs.append(
            Message("user", f"Your previous attempt was rejected:\n{feedback}\nFix it. JSON only.")
        )
    return msgs


def parse_code(reply: str) -> tuple[str, str]:
    obj = extract_json(reply)
    q = obj.get("quantizer_py")
    t = obj.get("test_py", "")
    if not isinstance(q, str) or not q.strip():
        raise LLMError("coder reply has no quantizer_py")
    return q, str(t)


def code(llm: LLM, spec: Spec, *, max_attempts: int = 2) -> tuple[str, str, list[str]]:
    """Ask for code; re-ask once with the static-check violations if needed."""
    feedback: str | None = None
    transcript: list[str] = []
    last_problems: list[str] = []
    for _ in range(max_attempts):
        reply = llm.complete(build_coder_messages(spec, feedback), json_mode=True)
        transcript.append(reply)
        try:
            quantizer_py, test_py = parse_code(reply)
        except LLMError as exc:  # unparseable reply: worth one retry, small models do this
            last_problems = [str(exc)]
            feedback = f"Your reply could not be parsed: {exc}"
            continue
        problems = check_source(quantizer_py) + has_required_contract(quantizer_py)
        if (
            f'name = "{spec.quantizer_name}"' not in quantizer_py
            and f"name = '{spec.quantizer_name}'" not in quantizer_py
        ):
            problems.append(f'class attribute name must be "{spec.quantizer_name}"')
        if not problems:
            return quantizer_py, test_py, transcript
        last_problems = problems
        feedback = "\n".join(problems)
    raise LLMError("coder could not produce acceptable code: " + "; ".join(last_problems))
