"""Build Phase 1 SRT (Self-Revision Training) dataset for TopoSD-Zero.

Pipeline:
  For each (x, a) in input:
    1. Sample y_init ~ pi_theta(.|x)   (on-policy)
    2. Score: r_out, r_topo, r_cont
    3. Build topology-aware P_r (dispatch over r_out x r_topo)
    4. Sample y_revised ~ pi_theta(.|x, y_init, P_r)
    5. Keep iff r_out(y_revised)=1 AND format_ok(y_revised)

Output: data/srt_ready/train.jsonl with records
  {messages: [...], y_init: str, P_r: str, y_revised: str,
   r_out: 0/1, r_topo: float, r_cont: float, orphan_step: int|null}
"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

# Note: actual model inference is performed by a separate rollout script
# (scripts/rollout_srt.py). This module builds dispatch + filtering logic.


REVISION_PROMPTS = {
    (1, "good"): "Let me rephrase the above solution more concisely.",
    (1, "bad"):  "Your answer is correct but step {k} has no dependency to earlier steps. Rewrite with explicit references.",
    (0, "good"): "Your reasoning looks well-structured but the final verdict is wrong. Let me reconsider.",
    (0, "bad"):  "Wait, this response is not correct, let me start over.",
}


@dataclass
class SRTRecord:
    problem: str
    solution: str
    y_init: str
    P_r: str
    y_revised: Optional[str] = None
    r_out_init: int = 0
    r_topo_init: float = 0.0
    r_cont_init: float = 0.0
    r_out_revised: int = 0
    format_ok_revised: bool = False
    orphan_step: Optional[int] = None
    structure_bucket: str = "bad"  # "good" if r_topo >= threshold else "bad"
    keep: bool = False


def build_prompt_dispatch(
    r_out: int,
    r_topo: float,
    topo_threshold: float = 0.5,
    orphan_step: Optional[int] = None,
) -> tuple[str, str]:
    """Return (bucket, P_r) per Eq.~topo_pr.

    bucket in {"good", "bad"} indicates whether r_topo >= threshold.
    """
    bucket = "good" if r_topo >= topo_threshold else "bad"
    template = REVISION_PROMPTS[(int(r_out), bucket)]
    k = orphan_step if orphan_step is not None else 1
    P_r = template.format(k=k)
    return bucket, P_r


def format_ok(text: str) -> bool:
    """Check if completion has a closed <answer>...</answer> tag."""
    return bool(re.search(r"<answer>.*?</answer>", text, re.DOTALL))


def filter_records(records: list[SRTRecord]) -> list[SRTRecord]:
    kept = []
    for r in records:
        if r.r_out_revised == 1 and r.format_ok_revised:
            r.keep = True
            kept.append(r)
    return kept


def to_training_example(r: SRTRecord) -> dict[str, Any]:
    """Convert kept record to an SFT-compatible training example.

    The resulting example trains two losses jointly:
      L_revision:  context = (x, y_init, P_r), target = y_revised
      L_generation: context = (x),             target = [y_init, P_r, y_revised]

    For simplicity we store both variants as separate training examples,
    with a `loss_type` tag that MS-Swift can use to multiply them by the
    corresponding loss coefficient.
    """
    problem = r.problem
    # Example A: revision task
    ex_revision = {
        "messages": [
            {"role": "user", "content": problem},
            {"role": "assistant", "content": r.y_init},
            {"role": "user", "content": r.P_r},
            {"role": "assistant", "content": r.y_revised or ""},
        ],
        "loss_type": "revision",
        "metadata": {
            "r_out_init": r.r_out_init,
            "r_topo_init": r.r_topo_init,
            "r_cont_init": r.r_cont_init,
            "bucket": r.structure_bucket,
            "orphan_step": r.orphan_step,
        },
    }
    # Example B: generation task (trains on y_init + P_r + y_revised as full continuation)
    ex_generation = {
        "messages": [
            {"role": "user", "content": problem},
            {
                "role": "assistant",
                "content": (r.y_init or "")
                + "\n\n" + r.P_r + "\n\n"
                + (r.y_revised or ""),
            },
        ],
        "loss_type": "generation",
        "metadata": ex_revision["metadata"],
    }
    return {"revision": ex_revision, "generation": ex_generation}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_rollouts", type=Path, required=True,
                    help="JSONL with rows: {problem, solution, y_init, y_revised, r_out_init, r_topo_init, r_cont_init, r_out_revised, orphan_step}")
    ap.add_argument("--output", type=Path, default=Path("data/srt_ready/train.jsonl"))
    ap.add_argument("--topo_threshold", type=float, default=0.5)
    ap.add_argument("--max_samples", type=int, default=0)
    args = ap.parse_args()

    records = []
    with args.raw_rollouts.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            bucket, P_r = build_prompt_dispatch(
                int(d.get("r_out_init", 0)),
                float(d.get("r_topo_init", 0.0)),
                args.topo_threshold,
                d.get("orphan_step"),
            )
            rec = SRTRecord(
                problem=d["problem"],
                solution=d.get("solution", ""),
                y_init=d["y_init"],
                P_r=P_r,
                y_revised=d.get("y_revised"),
                r_out_init=int(d.get("r_out_init", 0)),
                r_topo_init=float(d.get("r_topo_init", 0.0)),
                r_cont_init=float(d.get("r_cont_init", 0.0)),
                r_out_revised=int(d.get("r_out_revised", 0)),
                format_ok_revised=format_ok(d.get("y_revised", "") or ""),
                orphan_step=d.get("orphan_step"),
                structure_bucket=bucket,
            )
            records.append(rec)

    kept = filter_records(records)
    print(f"Input rollouts: {len(records)}, kept: {len(kept)} (after format+correctness filter)")

    if args.max_samples > 0:
        kept = kept[: args.max_samples]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        for r in kept:
            exs = to_training_example(r)
            f.write(json.dumps(exs["revision"], ensure_ascii=False) + "\n")
            f.write(json.dumps(exs["generation"], ensure_ascii=False) + "\n")

    print(f"Wrote {2 * len(kept)} training examples to {args.output}")


if __name__ == "__main__":
    main()
