# TopoPRM GRPO Reward Collapse Diagnosis (2026-04-20)

This report examines the three TopoPRM GRPO runs that drive our main-table
TopoPRM rows, reading raw `trainer_state.json` from each whitelisted
checkpoint. It is what we cite in the paper Limitations and what we use to
scope the follow-up roadmap.

## Runs inspected

| Tag | Checkpoint | Steps | Source |
|-----|-----------|------:|--------|
| hier-9b-v2 | `output/grpo_hierarchical_qwen35_9b_mcl4096/v2-20260407-162048/checkpoint-79` | 79 | 9B TopoPRM hierarchical |
| gated-9b-v4 | `output/grpo_gated_qwen35_9b_mcl4096/v4-20260407-111747/checkpoint-79` | 79 | 9B TopoPRM gated |
| qwen25_7b_hier-v3 | `output/grpo_hierarchical_qwen25_7b/v3-20260406-134423/checkpoint-318` | 318 | Qwen2.5-7B TopoPRM hierarchical |

## Reward trajectory summary

### hier-9b-v2 (step 1 -> 75)
| step | reward | reward_std | frac_zero_std | kl | grad_norm |
|---:|---:|---:|---:|---:|---:|
| 1  | 0.117 | 0.138 | 0.375 | 1.92 | 15.95 |
| 20 | 0.079 | 0.074 | 0.363 | 0.36 | 0.33 |
| 40 | 0.074 | 0.054 | 0.413 | 0.31 | 0.10 |
| 60 | 0.096 | 0.081 | 0.413 | 0.31 | 0.12 |
| 75 | 0.107 | 0.092 | 0.363 | 0.34 | 0.13 |

Observations:
- Reward mean drops from 0.12 to 0.07 in the first 20 steps, then oscillates 0.07-0.11. **No sustained upward trend.**
- `reward_std` monotonically shrinks in the first half and settles around 0.08.
- `frac_reward_zero_std` is persistently in the 0.35-0.45 band ? i.e. ~40% of each GRPO rollout group has zero reward variance, so those groups contribute no advantage signal.
- `grad_norm` is tiny (<0.5) after step 50, confirming the optimizer has almost nothing to move on.

Verdict: **stalled with local collapse**. Not fully dead, but not progressing.

### gated-9b-v4 (step 1 -> 75)
| step | reward | reward_std | frac_zero_std | kl | grad_norm |
|---:|---:|---:|---:|---:|---:|
| 1  | 0.0137 | 0.0004 | 0.188 | 1.92 | 15.90 |
| 20 | 0.0131 | 0.0006 | 0.100 | 0.49 | 0.25 |
| 40 | 0.0136 | 0.0006 | 0.113 | 0.37 | 0.13 |
| 60 | 0.0136 | 0.0006 | 0.175 | 0.34 | 0.32 |
| 75 | 0.0134 | 0.0005 | 0.225 | 0.50 | 9.19 |

Observations:
- **Gated reward is an order of magnitude smaller than hier** (?0.013 vs ?0.09). The gating threshold is so tight that almost no rollout crosses it.
- `reward_std ? 5e-4` for the entire 79 steps. This is essentially noise, not signal.
- The KL term and grad_norm hide this because the policy barely moves.

Verdict: **collapsed since step 1**. This directly explains why gated-v2 underperforms SFT on AIME/CNMO ? the GRPO stage never had a meaningful advantage to learn from; the adapter is effectively SFT + noise.

### qwen25_7b_hier-v3 (step 1 -> 315)
| step | reward | reward_std | frac_zero_std | kl | grad_norm |
|---:|---:|---:|---:|---:|---:|
| 1   | 0.224 | 0.006 | 0.00 | 0.62 | 0.37 |
| 50  | 0.225 | 0.011 | 0.00 | 0.64 | 0.44 |
| 100 | 0.224 | 0.010 | 0.00 | 0.60 | 0.43 |
| 200 | 0.225 | 0.009 | 0.00 | 0.59 | 0.43 |
| 300 | 0.225 | 0.011 | 0.00 | 0.61 | 0.46 |
| 315 | 0.224 | 0.010 | 0.00 | 0.58 | 0.45 |

Observations:
- Reward mean is frozen at 0.224 for all 318 steps. Std stays ~0.01, fracs ~0. `grad_norm` stable.
- In other words, the policy is in a plateau, not collapsing, but not learning.

Verdict: **stable plateau**. This is the least bad of the three, but still not progressing.

## Root causes

1. **Short training budget.** 79 steps on 9B GRPO is effectively warm-up. Advantage estimation needs more rollout diversity than 79 steps allow.
2. **Reward variance floor missing.** `frac_reward_zero_std` >0.3 (hier) and a `reward_std<1e-3` (gated) mean many rollouts in a group are indistinguishable. GRPO with zero-variance groups produces zero advantage; gradient direction comes from a minority of groups and is noisy.
3. **Gated reward threshold too tight for our 9B base.** The composite reward almost never crosses the gate, so the gated reward distribution is near-degenerate from step 1.

## Paper-side implications

- In `sections/6_appendix.tex` (Limitations), we cite these exact numbers to explain why TopoPRM pass@1 is not strictly above SFT on AIME/CNMO. The narrative in the main body remains: TopoPRM's contribution is structural/topology supervision, not an accuracy ceiling over SFT at short training budgets.
- `sections/3_method.tex` Stage II training details should add a sentence: "We report results at the short-run checkpoint (79 steps) that matches our current GRPO budget; Section (Limitations) documents the reward-variance floor that prevents continued improvement at this budget, which is left to future work."

## Actionable follow-ups (not executed in this sprint)

1. Continue training hier/gated from `checkpoint-79` to 300-500 steps with:
   - `reward_std_floor` regularization (penalize groups with `reward_std < 0.02`)
   - Gate threshold annealing (relax in early GRPO, tighten later)
2. Resample temperature for hier GRPO (currently 0.7; try 0.9 + longer `max_new_tokens=2560` during training to match eval-time budget)
3. Warm-start from the 7B hier-v3 318-step plateau with modified reward weighting to escape the plateau
