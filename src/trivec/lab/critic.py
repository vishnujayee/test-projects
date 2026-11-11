"""The Critic: rule-based checks first (cheap, deterministic), optional LLM code review second.

Verdicts: tampered | reject | suspicious | accept_unverified. The Critic never says "verified".
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from trivec.lab.llm import LLM, LLMError, Message, extract_json
from trivec.lab.runner import RunOutcome
from trivec.report import aggregate, load_results

EPS = 1e-6
SUSPICIOUS_GAIN = 0.02  # recall gain over the best baseline at <= bytes that demands a re-run


@dataclass
class Critique:
    verdict: str
    reasons: list[str] = field(default_factory=list)
    comparison: dict[str, Any] = field(default_factory=dict)
    llm_review: dict[str, Any] | None = None

    def to_markdown(self) -> str:
        lines = [f"# Critique\n\n**Verdict:** `{self.verdict}`\n"]
        if self.reasons:
            lines.append("\n".join(f"- {r}" for r in self.reasons))
        if self.comparison:
            lines.append("\n**Comparison with baselines (same dataset):**\n")
            lines.append("\n".join(f"- {k}: {v}" for k, v in self.comparison.items()))
        if self.llm_review:
            lines.append("\n**Code review (LLM):**\n")
            lines.append("\n".join(f"- {k}: {v}" for k, v in self.llm_review.items()))
        return "\n".join(lines) + "\n"


def baseline_rows(results_dir: Path, dataset: str, k: int) -> list[Any]:
    records = [r for r in load_results(results_dir) if r.get("source") != "lab"]
    groups = aggregate(records)
    for (ds, _model, kk), rows in groups.items():
        if ds == dataset and kk == k:
            return rows
    return []


def compare_to_baselines(record: dict[str, Any], results_dir: Path) -> dict[str, Any]:
    k = int(record.get("k", 10))
    rows = [
        r
        for r in baseline_rows(results_dir, record["dataset"]["name"], k)
        if r.mode == record["search_mode"]
    ]
    m = record["metrics"]
    recall, bytes_ = m[f"recall_at_{k}"], m["bytes_per_vector"]
    out: dict[str, Any] = {
        "recall": round(recall, 4),
        "bytes_per_vector": bytes_,
        "n_baselines": len(rows),
    }
    if not rows:
        return out
    f32 = next((r for r in rows if r.method == "float32"), None)
    if f32:
        out["float32_recall"] = round(f32.metrics[f"recall_at_{k}"][0], 4)
    cheaper_or_equal = [r for r in rows if r.metrics["bytes_per_vector"][0] <= bytes_ + EPS]
    if cheaper_or_equal:
        best = max(cheaper_or_equal, key=lambda r: r.metrics[f"recall_at_{k}"][0])
        out["best_baseline_at_or_below_bytes"] = best.method
        out["best_baseline_recall"] = round(best.metrics[f"recall_at_{k}"][0], 4)
        out["recall_gain_vs_best"] = round(recall - best.metrics[f"recall_at_{k}"][0], 4)
    return out


def rule_based(outcome: RunOutcome, results_dir: Path) -> Critique:
    if outcome.stage == "tampered":
        return Critique(
            "tampered",
            [
                "evaluator hash mismatch: "
                + "; ".join(outcome.eval_problems_before + outcome.eval_problems_after)
            ],
        )
    if outcome.stage == "static_check":
        return Critique("reject", ["static check failed", *outcome.static_violations])
    if outcome.stage == "tests":
        return Critique("reject", ["agent tests failed", outcome.tests_output[-800:]])
    rec = outcome.record or {}
    if rec.get("status") != "ok":
        err = rec.get("error") or {}
        return Critique(
            "reject",
            [f"evaluation failed: {err.get('type')}: {err.get('message')}", outcome.message],
        )

    reasons: list[str] = []
    k = int(rec.get("k", 10))
    m = rec["metrics"]
    integrity = rec.get("integrity") or {}
    if integrity.get("deterministic") is False:
        reasons.append("non-deterministic: two builds with the same seed produced different codes")
    if integrity.get("decode_follows_codes") is False:
        reasons.append("decode ignores its input: reordering the codes did not reorder the output")
    if integrity.get("state_growth_ok") is False:
        growth = integrity.get("state_growth_bytes_per_vector")
        reasons.append(
            f"hidden state grows with the corpus by {growth} bytes per vector: "
            "the real memory cost is not in bytes_per_vector"
        )
    if m[f"recall_at_{k}"] > 1.0 + EPS:
        reasons.append("impossible recall > 1.0")
    if abs(m["bytes_per_vector"] - m["bytes_per_vector_measured"]) > EPS:
        reasons.append(
            f"bytes_per_vector {m['bytes_per_vector']} != measured {m['bytes_per_vector_measured']}"
        )
    comparison = compare_to_baselines(rec, results_dir)
    if "float32_recall" in comparison and m[f"recall_at_{k}"] > comparison["float32_recall"] + EPS:
        reasons.append(
            "recall above the float32 reference on the same data: impossible, likely a bug"
        )
    if reasons:
        return Critique("reject", reasons, comparison)

    if comparison.get("recall_gain_vs_best", 0.0) > SUSPICIOUS_GAIN:
        return Critique(
            "suspicious",
            [
                f"beats the best baseline at the same or fewer bytes by {comparison['recall_gain_vs_best']:.3f} recall on ONE dataset and ONE seed; needs new seeds and a held-out dataset before anyone believes it"
            ],
            comparison,
        )
    return Critique(
        "accept_unverified", ["all rule checks passed; result recorded as unverified"], comparison
    )


CRITIC_SYSTEM = """You are the Critic in TriVec Lab. Review agent-written quantizer code for: data leakage
(anything that uses queries or labels), evaluator tampering, hidden randomness, wrong bytes accounting,
and whether the code actually implements the spec. Reply with ONE JSON object:
{"implements_spec": true|false, "leakage_risk": "low"|"medium"|"high", "concerns": ["..."], "summary": "..."}"""


def llm_review(llm: LLM, spec_md: str, quantizer_py: str) -> dict[str, Any]:
    msgs = [
        Message("system", CRITIC_SYSTEM),
        Message(
            "user", f"## Spec\n{spec_md}\n\n## Code\n```python\n{quantizer_py}\n```\nJSON only."
        ),
    ]
    try:
        return extract_json(llm.complete(msgs, json_mode=True))
    except LLMError as exc:
        return {"error": str(exc)}


def critique(
    outcome: RunOutcome,
    results_dir: Path,
    *,
    llm: LLM | None = None,
    spec_md: str = "",
    quantizer_py: str = "",
) -> Critique:
    c = rule_based(outcome, results_dir)
    if llm is not None and c.verdict in ("accept_unverified", "suspicious") and quantizer_py:
        c.llm_review = llm_review(llm, spec_md, quantizer_py)
        if (
            c.llm_review.get("leakage_risk") == "high"
            or c.llm_review.get("implements_spec") is False
        ):
            c.verdict = "suspicious"
            c.reasons.append("LLM code review raised a red flag (see below)")
    return c
