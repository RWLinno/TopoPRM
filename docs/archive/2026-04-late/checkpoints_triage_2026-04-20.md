# Checkpoint Triage Report (2026-04-20)

Scanned `output` for `**/checkpoint-*/trainer_state.json` -> 30 checkpoints.

## Summary by verdict

| Verdict | Count | Total size (MB) |
|---------|------:|----------------:|
| healthy | 6 | 10418 |
| stalled | 6 | 10615 |
| collapsed | 18 | 31508 |
| broken | 0 | 0 |

## Classification rules

- healthy: reward has upward trend, reward_std > 0.05, frac_reward_zero_std <= 0.3
- stalled: reward flat (|delta|<0.02), or frac_reward_zero_std > 0.3, or undertrained (<150 steps)
- collapsed: reward_std < 0.02 or frac_reward_zero_std > 0.8
- broken: grad_norm > 100, kl > 50, or no reward entries

## Detailed rows

| Verdict | Steps | Size(MB) | reward_first | reward_last | reward_max | reward_std_last | frac_zero_std | grad_norm_last | kl_last | Path | Reason |
|--------|------:|---------:|-------------:|------------:|----------:|----------------:|--------------:|---------------:|--------:|------|--------|
| collapsed | 79 | 1982 | 0.014 | 0.013 | 0.014 | 0.001 | 0.204 | 3.15 | 0.40 | `output/grpo_gated_qwen35_9b_mcl4096/v4-20260407-111747/checkpoint-79` | reward_std=0.0006 near 0 |
| collapsed | 79 | 1982 | 0.045 | 0.044 | 0.046 | 0.006 | 0.021 | 0.22 | 0.00 | `output/grpo_no_continuity_qwen35_9b/v0-20260404-163247/checkpoint-79` | reward_std=0.0063 near 0 |
| collapsed | 79 | 1982 | 0.045 | 0.044 | 0.046 | 0.006 | 0.013 | 0.22 | 0.00 | `output/grpo_no_topo_qwen35_9b/v0-20260404-162835/checkpoint-79` | reward_std=0.0061 near 0 |
| collapsed | 79 | 1982 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 7.59 | 5.55 | `output/grpo_outcome_only_qwen35_9b_mcl4096/v1-20260407-191217/checkpoint-79` | reward_std=0.0000 near 0 |
| collapsed | 79 | 1982 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.00 | 0.00 | `output/grpo_outcome_only_qwen35_9b/v2-20260404-012842/checkpoint-79` | reward_std=0.0000 near 0 |
| collapsed | 50 | 1982 | 0.014 | 0.013 | 0.014 | 0.001 | 0.183 | 0.13 | 0.36 | `output/grpo_gated_qwen35_9b_mcl4096/v4-20260407-111747/checkpoint-50` | reward_std=0.0005 near 0 |
| collapsed | 50 | 1982 | 0.045 | 0.043 | 0.046 | 0.006 | 0.025 | 0.22 | 0.00 | `output/grpo_no_continuity_qwen35_9b/v0-20260404-163247/checkpoint-50` | reward_std=0.0062 near 0 |
| collapsed | 50 | 1982 | 0.045 | 0.043 | 0.046 | 0.006 | 0.025 | 0.22 | 0.00 | `output/grpo_no_topo_qwen35_9b/v0-20260404-162835/checkpoint-50` | reward_std=0.0059 near 0 |
| collapsed | 50 | 1982 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.15 | 0.33 | `output/grpo_outcome_only_qwen35_9b_mcl4096/v1-20260407-191217/checkpoint-50` | reward_std=0.0000 near 0 |
| collapsed | 50 | 1982 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.00 | 0.00 | `output/grpo_outcome_only_qwen35_9b/v2-20260404-012842/checkpoint-50` | reward_std=0.0000 near 0 |
| collapsed | 318 | 1849 | 0.224 | 0.225 | 0.227 | 0.010 | 0.000 | 0.40 | 0.53 | `output/grpo_hierarchical_qwen25_7b/v3-20260406-134423/checkpoint-318` | reward_std=0.0097 near 0 |
| collapsed | 200 | 1848 | 0.224 | 0.226 | 0.227 | 0.010 | 0.000 | 0.45 | 0.57 | `output/grpo_hierarchical_qwen25_7b/v2-20260405-225622/checkpoint-200` | reward_std=0.0098 near 0 |
| collapsed | 79 | 1332 | 0.021 | 0.023 | 0.023 | 0.006 | 0.329 | 0.08 | 0.00 | `output/grpo_hierarchical_qwen35_9b_mcl4096/v1-20260407-015029/checkpoint-79` | reward_std=0.0061 near 0 |
| collapsed | 79 | 1332 | 0.013 | 0.013 | 0.013 | 0.000 | 1.000 | 0.00 | 0.00 | `output/grpo_gated_qwen35_9b_mcl4096/v3-20260407-020847/checkpoint-79` | reward_std=0.0000 near 0 |
| collapsed | 79 | 1332 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.00 | 0.00 | `output/grpo_outcome_only_qwen35_9b_mcl4096/v0-20260407-020847/checkpoint-79` | reward_std=0.0000 near 0 |
| collapsed | 50 | 1332 | 0.021 | 0.022 | 0.023 | 0.005 | 0.388 | 0.06 | 0.00 | `output/grpo_hierarchical_qwen35_9b_mcl4096/v1-20260407-015029/checkpoint-50` | reward_std=0.0053 near 0 |
| collapsed | 50 | 1332 | 0.013 | 0.013 | 0.013 | 0.000 | 1.000 | 0.00 | 0.00 | `output/grpo_gated_qwen35_9b_mcl4096/v3-20260407-020847/checkpoint-50` | reward_std=0.0000 near 0 |
| collapsed | 50 | 1332 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.00 | 0.00 | `output/grpo_outcome_only_qwen35_9b_mcl4096/v0-20260407-020847/checkpoint-50` | reward_std=0.0000 near 0 |
| healthy | 626 | 1994 | - | - | - | - | - | 0.27 | - | `output/sft_qwen35_9b/v0-20260407-011328/checkpoint-626` | sft steps=626, loss_last=0.3056, grad_norm_last=0.27 |
| healthy | 600 | 1982 | - | - | - | - | - | 0.29 | - | `output/sft_qwen35_9b/v0-20260407-011328/checkpoint-600` | sft steps=600, loss_last=0.2782, grad_norm_last=0.29 |
| healthy | 550 | 1982 | - | - | - | - | - | 0.33 | - | `output/sft_qwen35_9b/v0-20260407-011328/checkpoint-550` | sft steps=550, loss_last=0.2874, grad_norm_last=0.33 |
| healthy | 2034 | 1487 | - | - | - | - | - | 0.96 | - | `output/sft_distill_4b/v0-20260417-121952/checkpoint-2034` | sft steps=2034, loss_last=0.2650, grad_norm_last=0.96 |
| healthy | 2000 | 1487 | - | - | - | - | - | 0.90 | - | `output/sft_distill_4b/v0-20260417-121952/checkpoint-2000` | sft steps=2000, loss_last=0.2812, grad_norm_last=0.90 |
| healthy | 1900 | 1487 | - | - | - | - | - | 0.92 | - | `output/sft_distill_4b/v0-20260417-121952/checkpoint-1900` | sft steps=1900, loss_last=0.2635, grad_norm_last=0.92 |
| stalled | 79 | 1982 | 0.117 | 0.086 | 0.148 | 0.080 | 0.321 | 0.25 | 0.32 | `output/grpo_hierarchical_qwen35_9b_mcl4096/v2-20260407-162048/checkpoint-79` | frac_zero_std=0.32; undertrained steps=79 |
| stalled | 79 | 1982 | 0.154 | 0.151 | 0.179 | 0.045 | 0.113 | 0.18 | 0.37 | `output/grpo_no_topo_qwen35_9b_mcl4096/v1-20260407-191217/checkpoint-79` | reward trend flat (-0.003); undertrained steps=79 |
| stalled | 50 | 1982 | 0.117 | 0.080 | 0.148 | 0.063 | 0.404 | 1.55 | 0.32 | `output/grpo_hierarchical_qwen35_9b_mcl4096/v2-20260407-162048/checkpoint-50` | frac_zero_std=0.40; undertrained steps=50 |
| stalled | 50 | 1982 | 0.154 | 0.155 | 0.179 | 0.044 | 0.083 | 0.20 | 0.42 | `output/grpo_no_topo_qwen35_9b_mcl4096/v1-20260407-191217/checkpoint-50` | reward trend flat (+0.001); undertrained steps=50 |
| stalled | 79 | 1356 | 0.076 | 0.075 | 0.085 | 0.034 | 0.425 | 0.08 | 0.00 | `output/grpo_no_topo_qwen35_9b_mcl4096/v0-20260407-015238/checkpoint-79` | reward trend flat (-0.001); frac_zero_std=0.42; undertrained steps=79 |
| stalled | 50 | 1332 | 0.076 | 0.068 | 0.085 | 0.027 | 0.521 | 0.07 | 0.00 | `output/grpo_no_topo_qwen35_9b_mcl4096/v0-20260407-015238/checkpoint-50` | reward trend flat (-0.009); frac_zero_std=0.52; undertrained steps=50 |
