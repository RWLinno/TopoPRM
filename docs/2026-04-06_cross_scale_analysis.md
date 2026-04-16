# TopoPRM Cross-Scale Ablation Analysis
**Date**: 2026-04-06  
**Author**: Auto-generated from experiment results  
**Status**: All 14/14 experiments completed and evaluated

---

## 1. Executive Summary

TopoPRM's topology-aware process reward produces strong ablation evidence on Qwen3-32B (+13.0 vs Outcome Only), but the signal **completely fails to transfer** to Qwen3.5-9B and Qwen2.5-7B. On smaller models, removing topology actually *improves* performance. Three distinct failure modes are identified, each pointing to a specific fix.

---

## 2. Complete Results

### 2.1 Cross-Scale Ablation Table

| Config | 32B-Mid | 32B-High | 32B-Avg | 9B-Mid | 9B-High | 9B-Avg | 7B-Mid | 7B-High | 7B-Avg |
|--------|---------|----------|---------|--------|---------|--------|--------|---------|--------|
| TopoPRM (full) | **0.330** | **0.255** | **0.292** | 0.305 | 0.270 | 0.287 | 0.140 | 0.045 | 0.092 |
| w/o Topology | 0.230 | 0.165 | 0.198 | **0.340** | 0.280 | **0.310** | 0.210 | 0.050 | 0.130 |
| w/o Continuity | 0.275 | 0.200 | 0.238 | 0.295 | **0.290** | 0.292 | 0.120 | 0.080 | 0.100 |
| Outcome Only | 0.195 | 0.130 | 0.163 | 0.315 | 0.250 | 0.282 | **0.220** | **0.080** | **0.150** |
| TopoPRM ng4 | ? | ? | ? | **0.330** | **0.290** | **0.310** | ? | ? | ? |

### 2.2 Ablation Deltas (? vs TopoPRM full)

| Config | 32B ? | 9B ? | 7B ? |
|--------|-------|------|------|
| w/o Topology | **-9.5** ? | +2.3 ? | +3.8 ? |
| w/o Continuity | **-5.5** ? | +0.5 ? | +0.8 ? |
| Outcome Only | **-13.0** ? | -0.5 ? | **+5.8** ? |

**Key finding**: On 32B, every component contributes positively (topology is the strongest single contributor at -9.5). On 9B/7B, the pattern **reverses** ? topology becomes harmful, and simpler rewards outperform.

### 2.3 Format Compliance

| Config | 32B-Fmt | 9B-Fmt | 7B-FmtM | 7B-FmtH |
|--------|---------|--------|---------|---------|
| TopoPRM | 0.91 | 0.86 | 0.380 | 0.230 |
| w/o Topology | 0.89 | 0.84 | 0.260 | 0.355 |
| w/o Continuity | 0.88 | 0.83 | 0.475 | 0.520 |
| Outcome Only | 0.87 | 0.83 | 0.195 | 0.165 |

7B format compliance is catastrophically low (19-52%), indicating the model cannot reliably produce the required output structure.

---

## 3. Training Dynamics Diagnostics

### 3.1 Signal Quality Table

| Model | R_mean | R_std | ZeroStd | Clip | KL | Eval-Avg | Diagnosis |
|-------|--------|-------|---------|------|----|----------|-----------|
| 9B hier (ng2) | 0.099 | 0.026 | 0.077 | 1.000 | 0.003 | 0.287 | Truncated + weak signal |
| 9B no_topo | 0.044 | 0.006 | 0.014 | 1.000 | 0.004 | 0.310 | Truncated + weak signal |
| 9B no_cont | 0.044 | 0.006 | 0.019 | 1.000 | 0.004 | 0.292 | Truncated + weak signal |
| 9B outcome | 0.000 | 0.000 | 1.000 | 1.000 | 0.000 | 0.282 | **No gradient at all** |
| 9B ng4 | 0.198 | 0.019 | 0.088 | 1.000 | 0.012 | 0.310 | Truncated + moderate signal |
| 7B hier | 0.225 | 0.010 | 0.017 | 0.461 | 0.586 | 0.092 | **Gradient but wrong direction** |
| 7B no_topo | 0.100 | 0.000 | 0.989 | 0.023 | 0.087 | 0.130 | **No gradient** |
| 7B no_cont | 0.100 | 0.000 | 0.991 | 0.025 | 0.082 | 0.100 | **No gradient** |
| 7B outcome | 0.000 | 0.000 | 1.000 | 0.034 | 0.056 | 0.150 | **No gradient** |

### 3.2 Metric Definitions

- **ZeroStd**: Fraction of batches where all generations received identical reward ? GRPO advantage = 0 ? no policy gradient
- **Clip**: Fraction of generations truncated at `max_completion_length` ? incomplete reasoning chains
- **R_std**: Within-batch reward standard deviation ? GRPO needs this > 0 to learn

---

## 4. Root Cause Analysis: Three Failure Modes

### 4.1 Failure Mode 1: Truncation Noise (9B)

**Symptom**: 9B Clip = 1.000 for ALL configs. Removing topology *improves* by +2.3.

**Mechanism**: All 9B generations hit `max_completion_length=1024`. The topology reward evaluates an **incomplete** reasoning DAG from truncated output. Incomplete DAGs produce noisy/wrong topology scores that actively mislead training. Removing topology removes this noise source.

**Evidence**: 
- 9B no_topo (0.310) > 9B full (0.287): removing topo helps
- 9B ng4 (0.310) = 9B no_topo (0.310): more generations compensate but don't fix the underlying issue

**Fix**: Truncation-robust topology scoring ? evaluate partial DAGs gracefully, or increase `max_completion_length`.

### 4.2 Failure Mode 2: Reward Variance Collapse (7B ablations)

**Symptom**: 7B no_topo/no_cont/outcome all have ZeroStd ? 1.0 and R_std = 0.000.

**Mechanism**: Qwen2.5-7B-Instruct generates near-identical outputs for all 4 generations in a batch. Ablation rewards use fixed-weight linear combinations, so identical outputs ? identical rewards ? zero GRPO advantage ? zero gradient. The model doesn't learn at all; any eval improvement comes purely from KL regularization toward the SFT prior.

**Evidence**:
- 7B outcome: R=0.000, ZeroStd=1.000, KL=0.056 ? model drifts toward SFT via KL only
- 7B no_topo: R=0.100, ZeroStd=0.989 ? 98.9% of batches have zero gradient

**Fix**: The hierarchical reward's batch-level min-max rescaling already solves this (ZeroStd=0.017), but the amplified signal points in the wrong direction (see FM3).

### 4.3 Failure Mode 3: Structural Reward Hacking (7B hierarchical)

**Symptom**: 7B hier has the LOWEST eval (0.092) despite being the ONLY config with non-zero gradient.

**Mechanism**: The hierarchical formula `R = R_base � (1 + ??scale(R_topo) + (1-?)?scale(R_cont))` rewards structural properties (DAG validity, step connectivity) independently of correctness. On 7B:
- The model learns to produce **longer, more structured** outputs (tok=970-986, clip=0.461)
- But these outputs are **less accurate** (eval=0.092, worst of all configs)
- The multiplicative structure amplifies topology/continuity bonuses even when `R_base` (outcome) is near zero
- This creates a **reward hacking** pathway: optimize structure, ignore correctness

**Evidence**:
- 7B hier: R=0.225, clip=0.461, eval=0.092 (highest reward, worst eval)
- 7B outcome: R=0.000, clip=0.034, eval=0.150 (zero reward, best eval)
- The model with the highest training reward has the worst test performance

**Fix**: Gate topology/continuity bonuses on outcome correctness ? only reward good structure when the answer is also correct.

---

## 5. The Fundamental Problem

**TopoPRM's topology reward measures STRUCTURAL properties of reasoning chains (DAG validity, step connectivity). These are NECESSARY but not SUFFICIENT for correctness.**

On strong models (32B), structural quality correlates with correctness because the model already reasons well ? better structure ? better reasoning ? better answers.

On weak models (7B/9B), this correlation breaks:
- The model can produce well-structured but WRONG reasoning
- Structural rewards dominate outcome rewards in the hierarchical formula
- This creates a reward hacking pathway: optimize structure, ignore correctness

**This is a general problem for any process reward model**: process signals must be coupled with outcome signals to avoid rewarding "beautiful but wrong" reasoning.

---

## 6. Proposed Fix: Outcome-Gated Topology Reward

### 6.1 Core Idea

Replace the multiplicative formula:
```
R = R_base � (1 + ??topo + (1-?)?cont)     [current]
```

With an outcome-gated formula:
```
R = R_outcome � (1 + gate(R_outcome) � (??topo + (1-?)?cont)) + w_f?format
```

Where `gate(R_outcome) = sigmoid(? � (R_outcome - threshold))`:
- When outcome is correct (R_outcome > threshold): gate ? 1, topology/continuity bonuses apply
- When outcome is wrong (R_outcome < threshold): gate ? 0, only outcome signal matters
- This prevents rewarding "well-structured but wrong" reasoning

### 6.2 Additional Fixes

1. **Truncation-robust topology**: When output is truncated (detected by hitting max_completion_length), reduce topology weight by `(1 - clip_ratio)` to avoid evaluating incomplete DAGs
2. **Adaptive temperature**: Scale reward differences by `1/max(R_std, ?)` to maintain gradient even when raw variance is low
3. **Increase max_completion_length**: From 1024 ? 2048 for 9B to reduce truncation

### 6.3 Expected Impact

| Fix | Target Failure Mode | Expected Effect |
|-----|---------------------|-----------------|
| Outcome gating | FM3 (reward hacking) | Prevent topology from rewarding wrong answers |
| Truncation robustness | FM1 (truncation noise) | Reduce noisy topology signal on 9B |
| Longer generation | FM1 (truncation) | Allow complete DAGs on 9B |
| Adaptive temperature | FM2 (variance collapse) | Maintain gradient on 7B ablations |

---

## 7. Experiment Completion Status

| Task | Status | Key Result |
|------|--------|------------|
| 32B main + ablation (4 configs) | ? | TopoPRM effective, clear ablation |
| 9B main + ablation (4 configs) | ? | TopoPRM ineffective, ablation reversed |
| 9B ng4 hyperparameter | ? | Partial recovery to 0.310 |
| 9B mem70 hyperparameter | ? | No significant difference |
| 7B SFT | ? | checkpoint-624 |
| 7B main + ablation (4 configs) | ? | TopoPRM harmful, all ablations reversed |
| 7B eval (all 4 configs) | ? | Complete |
| 9B ng4 eval | ? | mid=0.330, high=0.290, avg=0.310 |
| Distillation (8B RKL) | ? | avg=0.590, far exceeds all GRPO |
| **Total** | **14/14** | **All complete** |

---

## 8. Paper Impact

### What to write in the paper:
1. **32B results as main contribution** ? topology rewards work when model capacity is sufficient
2. **Cross-scale analysis as important finding** ? reveals capacity threshold for process rewards
3. **Training dynamics diagnostics** ? ZeroStd, Clip, R_std as diagnostic tools for GRPO
4. **Failure mode taxonomy** ? truncation noise, variance collapse, reward hacking
5. **Outcome-gated topology** ? proposed fix with experimental validation (pending)

### Tables updated:
- `tables/private_results.tex` ? added 9B ng4 and 7B rows
- `tables/ablation_cross_scale.tex` ? new cross-scale ablation table
- `sections/4_experiments.tex` ? added cross-scale transferability paragraph
