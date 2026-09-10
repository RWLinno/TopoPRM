#!/usr/bin/env python3
"""Write output/eval/<tag>_gsm8k_metrics.json and _math500_metrics.json from benchmark_light reports."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "output" / "eval"
LIGHT = EVAL / "benchmark_light"

# (output_tag, gsm8k_report, math500_report)
PAIRS: list[tuple[str, Path | None, Path | None]] = [
    (
        "our_baseline_qwen25_7b",
        LIGHT / "baseline_qwen25_7b_gsm8k_gpu5/native/20260323_210524/reports/Qwen2.5-7B-Instruct/gsm8k.json",
        LIGHT / "baseline_qwen25_math500_gpu7/native/20260323_211446/reports/Qwen2.5-7B-Instruct/math_500.json",
    ),
    (
        "our_baseline_llama31_8b",
        LIGHT / "baseline_llama31_gsm8k_gpu7_retry2/native/20260324_000933/reports/Llama-3.1-8B-Instruct/gsm8k.json",
        LIGHT / "baseline_llama31_math500_gpu7_retry2/native/20260324_003302/reports/Llama-3.1-8B-Instruct/math_500.json",
    ),
    (
        "our_baseline_qwen3_32b",
        LIGHT / "baseline_qwen3_32b_gsm8k_gpu4/native/20260327_181725/reports/Qwen3-32B/gsm8k.json",
        LIGHT / "baseline_qwen3_32b_math500_gpu4/native/20260327_194640/reports/Qwen3-32B/math_500.json",
    ),
    (
        "our_sft_32b",
        LIGHT / "sft_gsm8k_gpu4/native/20260324_003713/reports/Qwen3-32B/gsm8k.json",
        None,
    ),
    (
        "our_grpo_outcome_32b",
        LIGHT / "grpo_outcome_gsm8k_full_gpu6/native/20260328_102030/reports/Qwen3-32B/gsm8k.json",
        LIGHT / "grpo_outcome_math500_gpu7_full/native/20260326_145717/reports/Qwen3-32B/math_500.json",
    ),
    (
        "our_grpo_no_topo_32b",
        LIGHT / "grpo_no_topo_gsm8k_gpu0/native/20260324_133222/reports/Qwen3-32B/gsm8k.json",
        None,
    ),
    (
        "our_grpo_clipped_32b",
        LIGHT / "grpo_clipped_gsm8k_full_gpu7/native/20260328_102019/reports/Qwen3-32B/gsm8k.json",
        LIGHT / "grpo_clipped_math500_gpu3_full/native/20260327_234714/reports/Qwen3-32B/math_500.json",
    ),
    (
        "our_grpo_main_32b",
        LIGHT / "grpo_main_gsm8k_full_gpu5/native/20260328_102031/reports/Qwen3-32B/gsm8k.json",
        LIGHT / "grpo_main_math500_gpu6_retry/native/20260326_024018/reports/Qwen3-32B/math_500.json",
    ),
    (
        "our_distill_rkl_8b",
        LIGHT / "distill_rkl_8b_compact_gsm8k_full_gpu3/native/20260328_101907/reports/Qwen3-8B/gsm8k.json",
        LIGHT / "distill_rkl_8b_compact_math500/native/20260328_022633/reports/Qwen3-8B/math_500.json",
    ),
]


def _acc(report: Path) -> float | None:
    if not report.exists():
        return None
    obj = json.loads(report.read_text(encoding="utf-8"))
    return float(obj.get("score", 0.0))


def main() -> None:
    EVAL.mkdir(parents=True, exist_ok=True)
    for tag, gsm_p, math_p in PAIRS:
        if gsm_p:
            a = _acc(gsm_p)
            if a is not None:
                (EVAL / f"{tag}_gsm8k_metrics.json").write_text(
                    json.dumps({"accuracy": a, "source": str(gsm_p)}, indent=2) + "\n",
                    encoding="utf-8",
                )
        if math_p:
            a = _acc(math_p)
            if a is not None:
                (EVAL / f"{tag}_math500_metrics.json").write_text(
                    json.dumps({"accuracy": a, "source": str(math_p)}, indent=2) + "\n",
                    encoding="utf-8",
                )
    print(f"[export_benchmark_metric_json] wrote metrics under {EVAL}")


if __name__ == "__main__":
    main()
