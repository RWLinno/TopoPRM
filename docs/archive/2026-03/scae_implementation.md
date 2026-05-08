# SCAE Implementation Notes (TopoPRM)

## What is implemented

- Reward-level stratified clipping class: `TopoSCAEReward`
- Registered key: `topo_composite_scae`
- Config: `configs/grpo_scae.yaml`
- Baseline (no SCAE): `configs/grpo_main.yaml`

## Why reward-level

In this repository, ms-swift GRPO internals are not modified. The plugin entry
point is ORM reward output. Therefore SCAE is implemented as a stratified
reward shaping approximation:

1. Compute base reward (`TopoCompositeReward` by default)
2. Split samples by outcome correctness proxy (`outcome >= 0.66`)
3. Normalize rewards within each group separately
4. Clip correct-group values to non-negative range and wrong-group values to
   non-positive range

This preserves the key idea of accuracy-first optimization while remaining
compatible with stock ms-swift training.

## On/off experiment

```bash
# baseline (no SCAE)
bash scripts/run_grpo.sh grpo_main

# SCAE-style stratified shaping
bash scripts/run_grpo.sh grpo_scae
```

Compare:

- reward stability (variance of logged reward)
- reward hacking ratio (high process reward + low outcome)
- length drift
