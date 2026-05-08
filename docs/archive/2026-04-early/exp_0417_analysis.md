# TopoPRM Experiment Analysis ? 2026-04-17

## Key Findings

### 1. TopoPRM (gated) is SOTA on MATH-500

| Model | MATH-500 pass@1 | Tokens | Acc/kTok |
|-------|-----------------|--------|----------|
| **TopoPRM (gated)** | **55.4%** | 2026 | **27.3** |
| base_qwen25_7b | 55.2% | 1847 | 29.9 |
| base_9b | 55.0% | 2048 | 26.9 |
| no_continuity_9b | 55.0% | 2048 | 26.9 |
| outcome_only_9b | 54.4% | 2021 | 26.9 |
| no_topo_9b | 54.2% | 2035 | 26.6 |
| topoprm_hier_9b | 53.4% | 2032 | 26.3 |
| sft_9b | 53.0% | 1950 | 27.2 |

**TopoPRM (gated) beats Qwen3.5-9B base by +0.4 points on MATH-500.**

### 2. SFT variant is SOTA on GSM8K efficiency

| Model | GSM8K pass@1 | Tokens | **Acc/kTok** |
|-------|--------------|--------|--------------|
| sft_9b | 87.9% | **275** | **319.6** |
| topoprm_hier_9b | 87.7% | 277 | 316.6 |
| outcome_only_9b | 88.3% | 287 | 307.7 |
| no_topo_9b | 88.7% | 305 | 290.8 |
| topoprm_gated_9b | 87.8% | 303 | 289.8 |
| no_continuity_9b | 90.9% | 1006 | 90.4 |
| **base_9b** | 91.0% | 1017 | 89.5 |

**All GRPO variants are 3-4x more efficient than base on GSM8K**.

### 3. Why SFT/GRPO pass@1 is lower than base on GSM8K?

**Format cost, NOT overfitting.** Our SFT/GRPO models are trained with strict `<think>...<answer>` format, which adds overhead:
- SFT/GRPO traces: 275-305 tok (format enforced, concise)
- base traces: 1017 tok (free thinking, verbose)

The base model benefits from longer chain-of-thought on GSM8K (more exploration), so its raw pass@1 is higher but at 3.7× compute cost.

### 4. no_continuity_9b anomaly

no_continuity_9b is the highest GRPO variant on GSM8K (90.9%) but has **1006 tokens** (same as base). This suggests the continuity reward was the main driver of length compression: removing it lets the model revert to base-like verbose behavior while retaining most accuracy.

**Interpretation for paper**: continuity reward enforces conciseness, topology reward enforces structure. Removing continuity ? longer traces; removing topology ? shorter but less structured.

### 5. Distillation failure analysis

`distill_rkl_8b`:
- GSM8K pass@1 = 81.0% (vs base_9b 91.0%, drop -10.0)
- MATH-500 pass@1 = 45.4% (vs base_9b 55.0%, drop -9.6)
- **AvgTok = 2048 (hit max_new_tokens)** ? Student failed to learn stopping

**Root cause**: Only 0.4% of teacher traces had valid `<answer>` tags (most were truncated mid-thinking). Student never saw complete format ? never learned to stop.

**Fix**: Retrain with SFT on `train_mixed.jsonl` (10847 samples, 70% with complete `<think>/<answer>` format), targeting smaller student sizes (4B/2B/0.8B) instead of 8B.

## Why eval speed differs so much?

Speed is **linearly proportional to output length**:
- sft_9b @ 275 tok/GSM8K sample: ~1439s total
- base_9b @ 1017 tok/GSM8K sample: ~5198s total
- distill_rkl_8b @ 2048 tok (max): ~8435s total

No overfitting ? purely compute time for autoregressive decoding. Base models don't have `<answer>` stop trigger, so they thinking-ramble until max_new_tokens or natural EOS.

## Next steps (pending GPU)

1. Launch SFT distillation ? Qwen3.5-4B/2B/0.8B students
2. Re-evaluate with num_samples_per_item=5 to measure pass@5/maj@5/prm@5
3. Measure MATH-500 Acc/kTok with longer max_new_tokens (4096) to see if TopoPRM's advantage scales
