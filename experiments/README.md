# TopoPRM NeurIPS Experiments

## Overview

All experiments follow a three-phase pipeline:
1. **SFT** — Supervised fine-tuning on critique data (shared across experiments)
2. **GRPO** — Reinforcement fine-tuning with different reward configurations
3. **Eval** — Benchmark evaluation (MATH, GSM8K, CMATH, GaoKao)

Optional fourth phase:
4. **Distillation** — Reverse-KL compression of long reasoning chains

## Experiment Matrix

| Experiment ID | Reward Config | Purpose |
|---------------|---------------|---------|
| `main` | 0.4 outcome + 0.15 format + 0.2 topo + 0.15 continuity + 0.1 length | Full TopoPRM |
| `ablation_outcome_only` | 1.0 outcome | Baseline: outcome-only reward |
| `ablation_no_topo` | 0.55 outcome + 0.15 format + 0.2 continuity + 0.1 length | Ablate topology reward |
| `ablation_no_continuity` | 0.55 outcome + 0.15 format + 0.2 topo + 0.1 length | Ablate continuity reward |
| `ablation_no_format` | 0.45 outcome + 0.25 topo + 0.2 continuity + 0.1 length | Ablate format reward |
| `ablation_topo_only` | 1.0 topo | Topo reward only |
| `distill_7b` | — | Reverse-KL distill to Qwen3-7B |
| `distill_1.5b` | — | Reverse-KL distill to Qwen3-1.5B |

## Running

```bash
# Full pipeline
bash experiments/run_main.sh

# Individual ablations
bash experiments/run_ablation_outcome_only.sh
bash experiments/run_ablation_no_topo.sh
bash experiments/run_ablation_no_continuity.sh

# Distillation
bash experiments/run_distill.sh
```

## Expected Results Table

| Model | MATH | GSM8K | CMATH | GaoKao | Avg Len |
|-------|------|-------|-------|--------|---------|
| Qwen3-14B (base) | — | — | — | — | — |
| + SFT | — | — | — | — | — |
| + GRPO (outcome only) | — | — | — | — | — |
| + GRPO (TopoPRM full) | — | — | — | — | — |
| + GRPO (no topo) | — | — | — | — | — |
| + GRPO (no continuity) | — | — | — | — | — |
| Distill → 7B | — | — | — | — | — |
| Distill → 1.5B | — | — | — | — | — |
