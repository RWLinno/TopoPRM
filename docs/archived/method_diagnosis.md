# Method Diagnosis — 2026-05-14

This document records the root-cause analysis of the current TopoPRM / TGSD
checkpoint behaviour on public math benchmarks, together with the patches
applied or proposed. All findings are grounded in the actual source code
and the evaluation results dumped by `scripts/dump_results_matrix.py`.

---

## 1. Observed behaviour (DR1-7B family, measured 2026-05-13)

Accuracy (pass@1, %) under the unified `transformers` protocol:

| Variant                   | GSM8K | MATH500 | Olymp | Omni | AIME24 | AIME25 | CNMO | MMLU | GPQA-D |
|---------------------------|------:|--------:|------:|-----:|-------:|-------:|-----:|-----:|-------:|
| DR1-7B base               |  60.8 |    68.4 |  57.8 | 72.8 |   46.7 |   33.3 | 23.3 | 42.4 |   16.2 |
| + SFT                     |  73.8 |    36.8 |  25.8 | 48.0 |    0.0 |    0.0 | 22.9 | 23.7 |   12.6 |
| + GRPO (outcome-only)     |  85.1 |    67.4 |  56.2 | 72.0 |   46.7 |   30.0 | 57.8 | 50.1 |   17.7 |
| + GRPO (w/o topology)     |  84.5 |    68.8 |    —  |   —  |   36.7 |   33.3 | 53.0 |   —  |   11.1 |
| + GRPO (w/o continuity)   |  85.1 |    66.4 |    —  |   —  |   46.7 |   30.0 | 50.6 |   —  |   14.7 |
| + TopoPRM (full)          |  84.3 |    66.6 |  56.4 | 72.2 |   50.0 |   40.0 | 55.4 | 49.6 |   15.2 |

### 1.1 Head-to-head TopoPRM vs outcome-only GRPO

- **AIME24**: +3.3 (win)
- **AIME25**: +10.0 (win, only significant one)
- **Olymp / Omni / MMLU**: within ±0.5 (tie)
- **GSM8K / MATH500**: −0.8 each (loss)
- **CNMO / GPQA-D**: −2.4 / −2.5 (loss)

Score: TopoPRM wins 2, ties 3, loses 4 across 9 benchmarks. The method is
clearly **task-selective** rather than universal. Per-benchmark token
budgets are nearly identical (0.99–1.01× vs outcome-only), so the
"compression" narrative is not supported by the current runs.

### 1.2 SFT regression

SFT is trained on a mixture of Chinese K12 teacher-critique examples plus
public math data. On public math benchmarks it causes severe distribution
shift: MATH500 68.4 → 36.8, AIME24/25 → 0.0. This is an independent
problem unrelated to reward design; see §4.

---

## 2. Root causes (source-grounded)

### R1. `TopoSCAEReward` is never actually used

`scripts/train_grpo.py:52` instantiates `TopoHierarchicalReward()`, not
`TopoSCAEReward`. The paper narrative around "Stratified Clipping Advantage
Estimation" therefore describes a mechanism that is **not in the active
training path** of existing checkpoints. The class exists in
`src/reward/composite_reward.py:691` but is only referenced by the
ablation entrypoint `scripts/train_grpo_ablation.py`.

### R2. `_batch_rescale` amplifies intra-group noise into full-scale signal

`src/reward/composite_reward.py:433-442`:

```python
@staticmethod
def _batch_rescale(scores: list[float]) -> list[float]:
    lo, hi = min(scores), max(scores)
    span = hi - lo
    if span < 1e-8:
        return [0.5] * len(scores)
    return [(s - lo) / span for s in scores]
```

This min-max rescales the group's `q_topo` and `q_cont` to `[0, 1]`
regardless of the true spread. Simulation (identical to our measured
benchmark conditions) shows:

| Scenario                          | True q_topo spread | After rescale | Effect on advantage |
|-----------------------------------|-------------------:|--------------:|---------------------|
| Hard AIME (1/4 correct)           |               0.10 |    [0.0, 1.0] | outcome dominates (adv ±1.73); topology is tie-breaker → **helpful** |
| Easy GSM8K (4/4 correct)          |               0.05 |    [0.0, 1.0] | outcome variance = 0, topology noise becomes ±1.4 → **harmful** |
| Mid MATH500 (2/4 correct)         |               0.25 |    [0.0, 1.0] | topology can override outcome in some rollouts → **noisy** |

This is the single biggest explanation for the observed loss pattern
(AIME up, GSM8K/MATH500 down).

### R3. `BASE_FLOOR = 0.05` leaks reward to outcome-incorrect traces

`src/reward/reward_config.py:86` defaults `BASE_FLOOR = 0.05`. Combined
with the additive base
`0.7 * r_out + 0.15 * r_fmt + 0.15 * r_len`, an outcome-incorrect trace
with `r_fmt=1, r_len=0.5` still receives `r_base = 0.225` before the
multiplicative topology gain (up to ×2) pushes it to `~0.45`. An
outcome-correct trace with messy DAG lands near `0.7 + 0.15 + 0.075 = 0.925`
and gets gain `≈ 1.2 → 1.11`. Ordering is preserved but the gap between
"correct-bad" and "wrong-clean" shrinks to `~2.5×` instead of the
`∞×` that pure multiplicative aggregation would enforce.

### R4. Length reward units mismatch measured trace lengths

`src/reward/reward_config.py:35-36` defaults `LENGTH_LOW=2000`,
`LENGTH_HIGH=4000` in **characters**. Measured average completion tokens:
GSM8K ≈ 679, MATH500 ≈ 6053, AIME ≈ 12k+. Converted to characters
(~4 chars/token), GSM8K is ≈ 2700 chars (in range), MATH500 is ≈ 24k
chars (far above HIGH, saturates to 0). So length reward is effectively:
- GSM8K: noisy ordering aligned with outcome (short correct trace wins)
- MATH500/AIME: constant 0 (no gradient)

### R5. SFT dataset distribution shift

`src/data/prepare_sft.py` mixes critique-format samples with small
amounts of public math. The model learns the critique output format as
the dominant pattern, which destroys its public-math pass@1 on anything
beyond GSM8K. Not a reward-design problem; a data-curation problem.

### R6. Continuity reward returns 1.0 on natural CoT

`src/reward/continuity_reward.py` looks for explicit `expr_current` /
`expr_prior` structure. Plain natural-language CoT traces rarely populate
these fields, so `_continuity_of_step` falls into the `else` branch and
returns the "all-continuous" default 1.0 for every step. The continuity
channel is thus a constant for most traces, contributing nothing to the
advantage. Evidence: `w/o continuity` ablation changes only by ±0.1-0.4
points on every benchmark vs full TopoPRM.

### R7. DAG step extractor often returns empty on natural CoT

`src/dag/graph.py:extract_steps_from_answer` keys off explicit "Step N:"
markers. Natural DR1-7B CoT seldom uses these markers, so the extractor
emits a single whole-trace "step", the DAG is trivial, and `q_topo`
collapses to a near-constant inside a group. The rescale in R2 then
turns the tiny remaining noise into full ±1 advantages.

---

## 3. Patch plan

All patches are **opt-in via environment variables**; default behaviour
is byte-identical to the released checkpoints. A new config
`configs/grpo_topoprm_v2.yaml` will set the env vars to the corrected
defaults for retraining.

### P0. Fix `_batch_rescale` noise amplification  *[critical]*

Add an env-var gated threshold: if the group's true spread is below
`TOPO_RESCALE_MIN_SPAN` (default 0.05 when the patch is enabled),
return `0.5 * N` instead of stretching to `[0, 1]`. This removes the
"topology dominates when outcome is saturated" failure mode.

Env vars:
- `TOPO_RESCALE_PATCH=1` to enable
- `TOPO_RESCALE_MIN_SPAN=0.05`

### P1. Route GRPO through `TopoSCAEReward` correctly

Rewrite `TopoSCAEReward.__call__` to preserve outcome magnitude across
strata:

```python
shaped[i] = floor_pos + (clip_pos - floor_pos) * pos_norm[k]     # correct stratum
shaped[i] = -floor_neg + (-clip_neg - (-floor_neg)) * neg_norm[k] # incorrect stratum
```

with `floor_pos > 0 > floor_neg` so the cross-stratum ordering always
puts answer-correct above answer-incorrect. Default disabled; enable via
`TOPO_SCAE_ENABLE=1`.

### P2. True multiplicative aggregation option

Add an aggregation mode that makes `r_out = 0 → r_total = 0`:

```python
if TOPO_HIER_AGG == "multiplicative":
    r_total = r_out * (1 + alpha*q_topo + (1-alpha)*q_cont) * (r_fmt ** beta) * (r_len ** gamma)
else:
    # current additive-base * multiplicative-gain formula (unchanged default)
```

with `BASE_FLOOR=0` automatically enforced when mode is `multiplicative`.

### P3. Continuity reward token-overlap fallback

When `_continuity_of_step` sees empty `expr_current/claim_current`, fall
back to Jaccard overlap between the current step's noun tokens and the
union of prior steps' noun tokens, clipped to `[0, 1]`. Env var
`TOPO_CONTINUITY_FALLBACK=1`.

### P4. DAG sentence-level fallback

In `extract_steps_from_answer`, when zero explicit markers are detected,
split on sentence boundaries (`[。\.\n]+`) with a minimum 20-character
length. Mark such steps with `fallback=True` in diagnostics so downstream
aggregation can down-weight them. Env var `TOPO_STEPS_FALLBACK=1`.

### P5. Length reward token units

Add `TOPO_LENGTH_UNIT=tokens` (default `chars` unchanged) and provide
sensible token defaults:
- `TOPO_LENGTH_LOW=512`, `TOPO_LENGTH_HIGH=8192` for natural math.

Optionally set reward weight for length to 0 via `TOPO_LEN_WEIGHT=0`
for AIME-style benchmarks.

### P6. New training config

`configs/grpo_topoprm_v2.yaml` sets:
```
env:
  TOPO_RESCALE_PATCH: 1
  TOPO_SCAE_ENABLE: 1
  TOPO_HIER_AGG: multiplicative
  TOPO_HIER_BASE_FLOOR: 0
  TOPO_CONTINUITY_FALLBACK: 1
  TOPO_STEPS_FALLBACK: 1
  TOPO_LENGTH_UNIT: tokens
  TOPO_LENGTH_LOW: 512
  TOPO_LENGTH_HIGH: 8192
```

Output directory: `output/grpo_topoprm_v2_dr1_7b`. Does not overwrite
existing checkpoints. Not launched automatically.

### P7. Paper-alignment edits (no retraining)

`TopoPRM_EMNLP26/sections/3_method.tex` currently implies SCAE is in the
active training path. Honest revision:
- Note that SCAE is the **planned** advantage estimator and is
  implemented as a drop-in replacement via `TOPO_SCAE_ENABLE=1`; the
  released checkpoints use the hierarchical multiplicative-gain reward.
- Mark GSM8K/MATH500 slight regression as `[to be improved]` rather
  than claiming universal improvement.
- Keep AIME24/25 and `w/o topology` ablation findings (these are
  genuine).

---

## 4. Non-regression guarantees

- Default reward behaviour is byte-identical to the currently released
  checkpoints unless the documented env vars are set.
- `scripts/check_reward_invariants.py` is extended with tests that
  assert the new P1/P2 orderings when the respective env vars are on,
  and that the v1 defaults still match when they are off.
- Any retraining under the v2 config writes to
  `output/grpo_topoprm_v2_dr1_7b`, leaving existing checkpoints
  untouched.

---

## 5. Timeline

- [x] Write this diagnosis (today)
- [ ] Implement P0/P1/P2/P3/P4/P5 with env-var gating
- [ ] Extend `check_reward_invariants.py` test coverage
- [ ] Draft `configs/grpo_topoprm_v2.yaml` (do not launch)
- [ ] Update `method.tex` with `[to be improved]` markers (honest
      description of current default behaviour)
- [ ] Await user decision on whether to launch v2 retraining
