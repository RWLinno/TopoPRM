"""Build topology-guided teacher revisions for compact-student distillation."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


SYSTEM_PROMPT = (
    "You are a math reasoning assistant. Reason step by step inside "
    "<think>...</think>, then write Final answer: \\boxed{...}."
)

REVISION_PROMPTS = {
    "cycle": (
        "The support graph contains circular justification among steps {nodes}. "
        "Rewrite the solution so every premise is established before it is used. "
        "Keep the reasoning concise and end with a boxed answer."
    ),
    "backward": (
        "Step {source} depends on later step {target}. Rewrite the solution in a valid "
        "premise-to-conclusion order, preserving the necessary support and a boxed answer."
    ),
    "orphan": (
        "Step {step} concludes without recoverable support. Rewrite the solution so that "
        "the conclusion explicitly follows from earlier steps, and end with a boxed answer."
    ),
    "continuity": (
        "Step {step} introduces an unsupported transition. Rewrite the solution with the "
        "missing dependency made explicit, keeping it concise and ending with a boxed answer."
    ),
    "compact": (
        "Rewrite the solution more concisely without deleting any premise needed for the "
        "conclusion. Keep a valid <think> block and end with a boxed answer."
    ),
    "answer": (
        "The final answer is incorrect. Re-solve the problem with a supported, acyclic "
        "derivation in premise-to-conclusion order and end with a boxed answer."
    ),
}
GENERIC_REVISION_PROMPT = (
    "Review and rewrite the preceding solution so it is correct, clear, and concise. "
    "Preserve the reasoning needed for the conclusion and end with a boxed answer."
)
LENGTH_REVISION_PROMPT = (
    "Rewrite the preceding solution correctly in at most {token_budget} tokens. "
    "Keep only reasoning needed for the conclusion and end with a boxed answer."
)


@dataclass
class DistillationRecord:
    problem: str
    solution: str
    y_init: str
    P_r: str
    y_revised: str
    defect_type: str
    r_out_init: int
    r_out_revised: int
    q_topo_init: float
    q_topo_revised: float
    q_dir_init: float
    q_dir_revised: float
    q_acyc_init: float
    q_acyc_revised: float
    q_cont_init: float
    q_cont_revised: float
    revised_tokens: int
    format_ok_revised: bool
    keep: bool = False
    rejection_reason: str = ""
    record_id: str = ""


SRTRecord = DistillationRecord


def _first(value: Any) -> Any:
    if isinstance(value, list) and value:
        return value[0]
    return value


def derive_record_seed(seed: int, record_id: str, stream: str) -> int:
    digest = hashlib.sha256(f"{seed}:{record_id}:{stream}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**31)


def build_prompt_dispatch(
    r_out: int,
    r_topo: float = 0.0,
    topo_threshold: float = 0.5,
    orphan_step: Optional[int] = None,
    *,
    defect: Optional[dict[str, Any]] = None,
) -> tuple[str, str]:
    """Return a localized defect type and teacher revision instruction.

    The positional arguments remain compatible with the released dispatcher.
    Canonical distillation passes the full raw/projected-graph defect record.
    """
    defect = defect or {}
    cycles = defect.get("cycle_components") or []
    backward = defect.get("backward_edges") or []
    orphan = defect.get("orphan_steps") or []
    continuity = defect.get("continuity_breaks") or []

    if cycles:
        nodes = ", ".join(str(x) for x in _first(cycles))
        kind = "cycle"
        prompt = REVISION_PROMPTS[kind].format(nodes=nodes)
    elif backward:
        edge = _first(backward)
        source = edge.get("source") if isinstance(edge, dict) else edge[0]
        target = edge.get("target") if isinstance(edge, dict) else edge[1]
        kind = "backward"
        prompt = REVISION_PROMPTS[kind].format(source=source, target=target)
    elif orphan or orphan_step is not None:
        step = _first(orphan) if orphan else orphan_step
        kind = "orphan"
        prompt = REVISION_PROMPTS[kind].format(step=step)
    elif continuity:
        kind = "continuity"
        prompt = REVISION_PROMPTS[kind].format(step=_first(continuity))
    elif int(r_out) == 0:
        kind = "answer"
        prompt = REVISION_PROMPTS[kind]
    else:
        kind = "compact"
        prompt = REVISION_PROMPTS[kind]

    if int(r_out) == 0 and kind not in {"answer"}:
        prompt = "The final answer is also incorrect. " + prompt
    return kind, prompt


def build_revision_instruction(
    strategy: str,
    *,
    score: dict[str, Any],
    token_budget: int,
) -> tuple[str, str]:
    if strategy == "topology":
        return build_prompt_dispatch(
            score["r_out"], score["r_topo"], defect=score["defect"]
        )
    if strategy == "generic":
        return "generic", GENERIC_REVISION_PROMPT
    if strategy == "length":
        return "length", LENGTH_REVISION_PROMPT.format(token_budget=token_budget)
    if strategy == "static":
        return "static", ""
    raise ValueError(f"Unknown revision strategy: {strategy}")


def _has_closed_boxed(text: str) -> bool:
    for match in re.finditer(r"\\boxed\{", text):
        depth = 1
        for char in text[match.end():]:
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return True
    return False


def format_ok(text: str) -> bool:
    lowered = text.lower()
    return "<think>" in lowered and "</think>" in lowered and _has_closed_boxed(text)


def revision_rejection_reason(
    record: DistillationRecord,
    *,
    topo_threshold: float,
    token_budget: int,
    selection: str = "topology",
    tolerance: float = 1e-8,
) -> str:
    if selection not in {"topology", "basic"}:
        raise ValueError(f"Unknown selection contract: {selection}")
    if record.r_out_revised != 1:
        return "incorrect_answer"
    if not record.format_ok_revised:
        return "invalid_format"
    if selection == "topology":
        if record.q_topo_revised + tolerance < topo_threshold:
            return "topology_below_threshold"
        if record.q_dir_revised + tolerance < record.q_dir_init:
            return "direction_degraded"
        if record.q_acyc_revised + tolerance < record.q_acyc_init:
            return "acyclicity_degraded"
    if record.revised_tokens <= 0 or record.revised_tokens > token_budget:
        return "over_budget"
    return ""


def filter_records(
    records: list[DistillationRecord],
    *,
    topo_threshold: float = 0.5,
    token_budget: int = 1024,
    selection: str = "topology",
) -> list[DistillationRecord]:
    kept = []
    for record in records:
        record.rejection_reason = revision_rejection_reason(
            record,
            topo_threshold=topo_threshold,
            token_budget=token_budget,
            selection=selection,
        )
        record.keep = not record.rejection_reason
        if record.keep:
            kept.append(record)
    return kept


def to_training_example(record: DistillationRecord) -> dict[str, Any]:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": record.problem},
            {"role": "assistant", "content": record.y_revised},
        ],
        "record_id": record.record_id,
        "metadata": {
            "defect_type": record.defect_type,
            "q_topo_init": record.q_topo_init,
            "q_topo_revised": record.q_topo_revised,
            "q_dir_init": record.q_dir_init,
            "q_dir_revised": record.q_dir_revised,
            "q_acyc_init": record.q_acyc_init,
            "q_acyc_revised": record.q_acyc_revised,
            "revised_tokens": record.revised_tokens,
        },
    }


def _record_from_dict(row: dict[str, Any]) -> DistillationRecord:
    revised = str(row.get("y_revised", "") or "")
    return DistillationRecord(
        problem=str(row.get("problem", "")),
        solution=str(row.get("solution", "")),
        y_init=str(row.get("y_init", "")),
        P_r=str(row.get("P_r", "")),
        y_revised=revised,
        defect_type=str(row.get("defect_type", row.get("bucket", "unknown"))),
        r_out_init=int(row.get("r_out_init", 0)),
        r_out_revised=int(row.get("r_out_revised", 0)),
        q_topo_init=float(row.get("q_topo_init", row.get("r_topo_init", 0.0))),
        q_topo_revised=float(row.get("q_topo_revised", row.get("r_topo_revised", 0.0))),
        q_dir_init=float(row.get("q_dir_init", 0.0)),
        q_dir_revised=float(row.get("q_dir_revised", 0.0)),
        q_acyc_init=float(row.get("q_acyc_init", 0.0)),
        q_acyc_revised=float(row.get("q_acyc_revised", 0.0)),
        q_cont_init=float(row.get("q_cont_init", row.get("r_cont_init", 0.0))),
        q_cont_revised=float(row.get("q_cont_revised", row.get("r_cont_revised", 0.0))),
        revised_tokens=int(row.get("revised_tokens", 0)),
        format_ok_revised=bool(row.get("format_ok_revised", format_ok(revised))),
        record_id=str(row.get("record_id", "")),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_rollouts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validation_output", type=Path)
    parser.add_argument("--validation_fraction", type=float, default=0.05)
    parser.add_argument("--topo_threshold", type=float, default=0.5)
    parser.add_argument("--token_budget", type=int, default=1024)
    parser.add_argument("--selection", choices=["topology", "basic"], default="topology")
    parser.add_argument("--max_samples", type=int, default=0)
    args = parser.parse_args()

    records: list[DistillationRecord] = []
    with args.raw_rollouts.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(_record_from_dict(json.loads(line)))

    kept = filter_records(
        records,
        topo_threshold=args.topo_threshold,
        token_budget=args.token_budget,
        selection=args.selection,
    )
    eligible = len(kept)
    if args.max_samples > 0:
        kept.sort(
            key=lambda record: hashlib.sha256(
                (record.record_id or record.problem).encode("utf-8")
            ).hexdigest()
        )
        kept = kept[:args.max_samples]

    validation: list[DistillationRecord] = []
    train: list[DistillationRecord] = kept
    if args.validation_output:
        fraction = max(0.0, min(0.5, args.validation_fraction))
        train = []
        for record in kept:
            source_id = record.record_id.split(":sample-")[0] or record.problem
            bucket = int(hashlib.sha256(source_id.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
            (validation if bucket < fraction else train).append(record)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as handle:
        for record in train:
            handle.write(json.dumps(to_training_example(record), ensure_ascii=False) + "\n")
    if args.validation_output:
        args.validation_output.parent.mkdir(parents=True, exist_ok=True)
        with args.validation_output.open("w") as handle:
            for record in validation:
                handle.write(json.dumps(to_training_example(record), ensure_ascii=False) + "\n")

    rejected = Counter(r.rejection_reason for r in records if not r.keep)
    defects = Counter(r.defect_type for r in kept)
    print(json.dumps({
        "input": len(records),
        "eligible": eligible,
        "kept": len(kept),
        "train": len(train),
        "validation": len(validation),
        "acceptance_rate": eligible / max(len(records), 1),
        "selection": args.selection,
        "rejections": dict(sorted(rejected.items())),
        "accepted_defects": dict(sorted(defects.items())),
        "output": str(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
