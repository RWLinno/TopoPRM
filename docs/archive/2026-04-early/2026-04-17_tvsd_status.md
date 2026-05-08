# TVSD Status (2026-04-17)

## Executive Summary

Implemented the full TVSD pipeline on top of the existing TopoPRM
infrastructure: (1) upgraded eval protocol with chat template + SFT-style
system prompt, (2) added 10-benchmark coverage (GSM8K / MATH-500 / Olympiad
/ Omni-MATH / AIME'24 / AIME'25 / CNMO'24 / MMLU / GPQA-D; LiveCode dropped 2026-04-21) with
pass@k / maj@k / prm@k metrics, (3) implemented on-policy distillation-style Phase 1 SRT +
Phase 2 OPSD with topology-aware revision prompts, (4) rewrote paper �3.5,
�4.RQ4 and added unified tables, (5) pivoted student target from 8B RKL
(known failure) to Qwen3.5-{4B, 2B, 0.8B} OPSD.

## Key Result (Early)

With the new chat-template + SFT-style prompting protocol, **`sft_9b_v2`
reaches 96.1% on GSM8K after 76 items** (up from 87.9% under raw-text
prompt), crossing the 90% threshold. The v2 protocol finally lets our
SFT/GRPO variants express their true capability.

Model | Protocol | GSM8K
---   | ---      | ---
Qwen3.5-9B base | raw-text | 91.0
Qwen3.5-9B + SFT | raw-text | 87.9 (formatting cost)
**Qwen3.5-9B + SFT** | **chat+sft_style (v2)** | **96.1+ (preliminary)**

## What's running now (2026-04-17, 15:45 UTC)

- GPU 4: `topoprm_hier_9b_v2` ? chat+sft_style, 5-sample, GSM8K+MATH-500
- GPU 5: `sft_9b_v2` ? chat+sft_style, 5-sample, GSM8K+MATH-500 (at 76/1319)

Both started via `scripts/bench_transformers.py` with new flags:
`--use_chat_template --sft_style --num_samples_per_item 5 --k_values 1 5`.

## What's been built (code + docs)

### Evaluation infrastructure
- `scripts/bench_transformers.py` ? upgraded with:
  - Chat-template prompting (`--use_chat_template`)
  - SFT-style system prompt (`--sft_style`) matching training format
  - 3-shot GSM8K / 1-shot MATH few-shot (`--fewshot`)
  - Enhanced answer extraction (MCQ: "The answer is A", "(A)" etc.)
  - 10 benchmark loaders with local-first strategy
  - pass@1/pass@k/maj@k/prm@k/error/correct/F1/#Tokens
- `scripts/bench_gen_then_revise.py` ? on-policy distillation Generate-then-Revise mode
- `scripts/fill_rft_csv.py` ? aggregates eval JSONs into CSV template
- `scripts/run_extended_benchmarks.sh` ? queues AIME/CNMO/MMLU/GPQA runs

### TVSD pipeline
- `src/distill/build_srt_data.py` ? 2�2 dispatch (r_out � r_topo) for P_r
- `scripts/rollout_srt.py` ? Phase 1 on-policy rollout + score
- `src/distill/opsd_trainer.py` ? Phase 2 on-policy self-distillation
- Configs: `configs/srt_9b.yaml`, `configs/opsd_9b.yaml`,
  `configs/opsd_student_{4b,2b,0p8b}.yaml`

### Paper (NeurIPS submission)
- `sections/3_method.tex` ? **new �3.5 TVSD**:
  - Phase III-A SRT with Eq.~eq:topo_pr (topology-aware dispatch table)
  - Phase III-B OPSD with Eq.~eq:opsd
  - Discussion on why topology-aware P_r concentrates KL gradient
- `sections/4_experiments.tex` ? **RQ4 rewritten**:
  - From-plain-RKL-to-TVSD motivation (0.4% closed-answer rate)
  - Revision gain table reference
  - Compression narrative updated
- `tables/public_results_unified.tex` ? 10-benchmark pass@1 table
- `tables/unified_metrics.tex` ? full 8-column per-benchmark metrics
- `tables/revision_gain.tex` ? First-Attempt vs Revised Attempt
- `references.bib` ? +6 entries (on-policy distillation, OPD, OPSD, SDFT, DAPO, RFT)
- `proposal.md` ? updated model matrix + TVSD section

## Outstanding work (training pending, requires more GPU time)

1. Run `scripts/rollout_srt.py` to collect ~8k (x, y_init, P_r, y_revised)
   triples for SRT Phase 1.
2. `swift sft --config configs/srt_9b.yaml` ? Phase 1 SRT on 9B.
3. `python -m src.distill.opsd_trainer --config configs/opsd_9b.yaml` ? Phase 2 OPSD on 9B.
4. Student compression runs: `configs/opsd_student_{4b,2b,0p8b}.yaml`.
5. Full 10-benchmark evaluation of: base, sft_v2, topoprm_hier_v2, topoprm_gated_v2,
   srt_9b, opsd_9b, opsd_student_4b, opsd_student_2b, opsd_student_0p8b.

## What changed in evaluation protocol (v2 vs v1)

| Aspect | v1 (raw text) | v2 (chat + sft_style) |
|---|---|---|
| Base model prompt | `Question:{q}\n\nAnswer:` | `system` + `user` with SFT format |
| SFT/GRPO prompt | Same as base (mismatch) | `<think>/<answer>` system prompt |
| sft_9b GSM8K | 87.9% | **96.1+%** (preliminary) |
| Extraction priority | `####` > `\\boxed` > last num | `<answer>` > `\\boxed` > phrase > last num |
| MCQ extractor | Any single letter | "The answer is X", "(X)", "X)" patterns |
| Samples per item | 1 (greedy) | 5 (temp 0.7, k={1,5}) |
| Metrics reported | pass@1, tokens | 8 metrics � 10 benchmarks |
