# P0 Code & Claim Audit (TopoPRM rebuttal)

Date: 2026-07-09. Branch: `rebuttal`. Env: `/Knowin/foundation/weilinruan/env/topoprm/bin/python` (swift 4.2.0.dev0, torch 2.5.1+cu121, vllm 0.6.0, math_verify OK).

## A. Outcome reward on public benchmarks (TsKG W-comment, T4)

**Finding: public math benchmarks use EXACT ANSWER CORRECTNESS, not rubric score.**

- `src/reward/outcome_reward.py::OutcomeReward` extracts `\boxed{}` / last-numeric answer and verifies via `math_verify.verify` (symbolic equivalence), returning 1.0/0.0. No rubric anywhere in this path.
- The rubric-score reward is confined to the in-domain critique set (Appendix), where `Acc` = exact-match on an integer rubric score. It is never imported by the public-benchmark reward funcs.
- Ablation configs (`grpo_outcome_only_qwen35_9b.yaml`, `grpo_no_topo_*`, `grpo_no_continuity_*`) all route through `ablation_rewards.py`, whose `OutcomeOnlyReward` wraps `OutcomeReward` directly.

Conclusion: the reviewer's worry ("if the rubric reward was used there, re-run") is a documentation-clarity problem, not a code problem. No public-benchmark rerun is required on reward-correctness grounds.

## B. Hierarchical reward = correctness-first multiplicative (B5w7 W2, TsKG W3)

`src/reward/composite_reward.py::TopoHierarchicalReward.__call__` (L479-526):
- multiplicative mode: `r = r_base_floored * (1 + alpha*scale(r_topo) + (1-alpha)*scale(r_cont))`, with `r_base = outcome * format_gate * length_gate`.
- Hard correctness primacy: when `outcome = 0`, `r_base = 0`, so no process bonus can lift a wrong trace above a correct one. Verified in code comment L510 + `tests/test_rewards.py`.
- `TOPO_HIER_BASE_FLOOR=0.05` only applies in additive mode to avoid zero-variance rollout groups; in multiplicative mode `r_base_floored = r_base` (no floor). This matches the released checkpoints.

## C. ACE vs SCAE naming (HxUk, TsKG both reference)

- Main text calls the estimator **ACE** (Asymmetric Clipping Advantage Estimator); `3_method_0520.tex` + appendix + `TopoSCAEReward` in code use **SCAE**. Same estimator.
- Released checkpoints use `TopoHierarchicalReward` (reward-level correctness-first shaping), NOT the reward-output SCAE variant (`SCAE_PRESERVE_OUTCOME` defaults OFF, L151 reward_config).
- Action for manuscript: unify to one name (ACE) and state that stratified clipping is realized through the multiplicative correctness-first reward + GRPO group normalization.

## D. Token-length inconsistency (TsKG W2, T2) — CONFIRMED, must fix text

Real generated-token counts from released eval JSONs (`rebuttal/outputs/hf_eval_results`):

| label / bench | pass@1 | avg_tokens |
| --- | ---: | ---: |
| 9b_v2_ckpt120 / omni_math | 0.514 | 4079.7 |
| 9b_v2_ckpt120 / aime2024 | 0.033 | 4096.0 (cap) |
| topoprm_q25_7b / math500 | 0.668 | 895.7 |
| topoprm_q25_7b / aime2024 | 0.133 | 1506.3 |
| topoprm_q25_7b / cnmo2024 | 0.470 | 1173.6 |

- The "<500 tokens" phrase in the training-dynamics figure text is NOT consistent with eval-time generation (895-4096 tokens). It refers to a different quantity (mean step/segment length during training rollouts under a tight budget), but the paper does not disambiguate.
- **Fix**: remove "<500 tokens"; report benchmark-specific eval token means and the "15-24% fewer tokens vs outcome-only GRPO" claim, which is supported by `ablation_reward.tex` / `efficiency.tex`.

## E. Table 1 provenance (TsKG W2, T3)

`tables/main_accuracy.tex` caption already states "collected from the original papers or reproduced under the same evaluation protocol" and separates a shaded "Reference open-source models (not bolded/underlined)" block from the matched backbone groups. Remaining risks flagged by TsKG:
- DR1-7B base MATH row (36.8) jumps to 92.8 after SFT: plausible only if the base row is greedy pass@1 while SFT row is pass@5 under boxed extraction — provenance must be labeled per row.
- Action: add checkpoint IDs + quoted/reproduced flag per row; the matched deltas we cite in the rebuttal are only within-backbone (outcome-only GRPO vs TopoPRM, same SFT ckpt, same 200 GRPO steps) from `ablation_reward.tex`.

## F. Data available for rebuttal experiments

- `data/grpo_ready/train_public.jsonl`: 19,472 records (gsm8k 7,472 + math 12,000), each with `reference_dag` = {nodes[step_id, exprs, claims, step_type, local_verdict], edges[source, target, edge_type, dep_type, weight]}. This is the ground-truth-ish extractor output usable for edge validation and semantic-gap analysis.
- Released eval JSONs for 9B (ckpt40/120) and q25-7b.
- HF ckpts available: `sft-dr1-7b-final`, `grpo-topoprm-dr1-7b`, `grpo-topoprm-qwen35-9b`, `opd-*`, `grpo-scae-qwen35-9b`.

## G. Eval-protocol fix (discovered during rebuttal reruns)

The released GRPO adapters (and our TRL-trained ones) were trained on
**base + SFT (merged)**, using the `deepseek_r1` `<think>/<answer>` template.
Evaluating the GRPO adapter alone on the raw base model with a generic prompt
gives badly understated numbers (e.g. full TopoPRM AIME 3.3%, GSM8K truncated at
the token cap). Correct protocol, now used by `rebuttal/scripts/run_matched_eval.sh`:
1. Merge base + SFT + GRPO into one model (`merge_stacked_adapter.py`).
2. Evaluate with `--sft_style` (the trained `<think>/<answer>` prompt).
3. Use a large token budget (>=4096 GSM8K, 8192 MATH/AIME).
Sanity check: outcome+length baseline GSM8K jumps 58% -> 76.5% (avg_tok 277,
confirming length control) under the corrected protocol.

## Prioritized next steps
1. LLM-based edge validation on ~120 sampled traces (all three W1). [P0]
2. Structure-semantic gap table from reference_dag + outcome (B5w7 W2, TsKG comment). [P0]
3. Outcome+length GRPO baseline train + eval (HxUk W2, B5w7 W3). [P1]
4. Non-Qwen (DR1-7B) sanity eval (HxUk W3). [P1]
