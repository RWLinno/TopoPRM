#!/usr/bin/env python3
"""Convert train_public.jsonl -> ms-swift GRPO jsonl (messages + solution + reference_dag)."""
import json
import sys
from pathlib import Path

SRC = Path("data/grpo_ready/train_public.jsonl")
DST = Path("data/grpo_ready/train_public_swift.jsonl")

SYSTEM = "Solve the math problem step by step. Put the final answer in \\boxed{}."


def main() -> None:
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else SRC
    dst = Path(sys.argv[2]) if len(sys.argv) > 2 else DST
    n = 0
    with src.open() as fin, dst.open("w") as fout:
        for line in fin:
            if not line.strip():
                continue
            d = json.loads(line)
            rec = {
                "messages": [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": d["question"]},
                ],
                "solution": str(d.get("final_answer", "")),
                "reference_dag": json.dumps(d.get("reference_dag", {}), ensure_ascii=False),
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    print(f"[to_swift_grpo] wrote {n} records -> {dst}")


if __name__ == "__main__":
    main()
