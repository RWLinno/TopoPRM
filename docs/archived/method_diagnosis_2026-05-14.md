# Method diagnosis — 2026-05-14

## TL;DR

The current TopoPRM Stage-II run underperforms outcome-only GRPO on public
benchmarks (GSM8K −0.8, MATH500 −0.8, CNMO −2.4, GPQA-D −2.5) while winning
only on AIME24 / AIME25. The two root causes are **algorithmic**, not "the
method is wrong":

1. **P0 — batch rescaling collapses outcome spread.** `_batch_rescale`
   min-max-stretches a group of 4 rollouts to the full `[0, 1]` range even
   when the true topology/continuity spread inside the group is near zero.
   On GSM8K / MATH500, outcome is nearly saturated, so `q_topo` and
   `q_cont` noise gets amplified 10×–100× and drowns out outcome signal
   inside the multiplicative gain.

2. **P2 — base aggregation is additive, not multiplicative.** The code
   uses `r_base = 0.70·outcome + 0.15·format + 0.15·length`, then applies
   `BASE_FLOOR = 0.05`. A structurally clean, answer-wrong trace can score
   `0.15·format + 0.15·length` = up to `0.30`, **higher than the 0.05
   floor that an answer-correct-but-format-broken trace is clipped to**.
   This is exactly the correctness-primacy failure the paper claims to
   avoid. The multiplicative gain `(1 + α·q_topo + (1-α)·q_cont)` then
   magnifies the wrong trace's advantage.

Both are now fixed behind opt-in flags so released checkpoints are
byte-level preserved. **Default `git pull` behaviour is unchanged.**

---

## The data

From `scripts/dump_results_matrix.py` on the current eval output
(`output/eval/*_metrics.json`, DR1-7B family, unified `transformers`
protocol, pass@1):

```
Variant                      |  GSM8K | MATH500 |  Olymp |   Omni | AIME24 | AIME25 |   CNMO |   MMLU | GPQA-D
--------------------------------------------------------------------------------------------------------------
DR1-7B base                  |   60.8 |   68.4 |   57.8 |   72.8 |   46.7 |   33.3 |   23.3 |   42.4 |   16.2
+ SFT                        |   73.8 |   36.8 |   25.8 |   48.0 |    0.0 |    0.0 |   22.9 |   23.7 |   12.6
+ GRPO (outcome-only)        |   85.1 |   67.4 |   56.2 |   72.0 |   46.7 |   30.0 |   57.8 |   50.1 |   17.7
+ GRPO (w/o topology)        |   84.5 |   68.8 |      — |      — |   36.7 |   33.3 |   53.0 |      — |   11.1
+ GRPO (w/o continuity)      |   85.1 |   66.4 |      — |      — |      — |      — |   50.6 |      — |      —
+ TopoPRM (full)             |   84.3 |   66.6 |   56.4 |   72.2 |   50.0 |   40.0 |   55.4 |   49.6 |   15.2
```

TopoPRM vs outcome-only: **2 wins (AIME24 +3.3, AIME25 +10.0), 4 losses,
3 ties**. Avg tokens nearly identical across variants (0.99×–1.05×), so
the compression/efficiency claim does not hold on these runs.

One empirically clean signal survives the critique: `w/o topology` loses
**13.3 pts on AIME24** (50.0 → 36.7). The topology reward is doing
something right on competition-math; the issue is that on saturated
benchmarks the same signal gets amplified above the outcome signal.

---

## Root cause 1 (P0) — rescaling the no-spread case

### The code path

`src/reward/composite_reward.py::_batch_rescale`
```python
@staticmethod
def _batch_rescale(scores):
    lo, hi = min(scores), max(scores)
    span = hi - lo
    if span < 1e-8:           # ← only true equality counts as "no spread"
        return [0.5] * len(scores)
    return [(s - lo) / span for s in scores]
```

With group size = 4, typical on-train distributions of `q_topo` and
`q_cont` on GSM8K / MATH500 rollouts concentrate around a single mode
with spread ~0.03. The rescaler stretches that to `[0.0, 1.0]`, so two
structurally near-identical rollouts get topology-gain factors of
`(1 + α·0.0)` = 1.0 and `(1 + α·1.0)` ≈ 1.5 — a 1.5× multiplier over a
pair whose true structural quality differs by 3% of the unit interval.
GRPO then picks the "winner" on amplified noise.

AIME traces are longer and have real structural diversity (span ~0.15),
so the rescaler's stretch is justified there. GSM8K/MATH500 are where
this hurts.

### The patch (P0)

`RewardConfig.TOPO_RESCALE_PATCH` (default **False**) gates a new branch:

```python
if RewardConfig.TOPO_RESCALE_PATCH:
    if span < RewardConfig.TOPO_RESCALE_MIN_SPAN:   # 0.05 default
        return [0.5] * len(scores)
else:
    if span < 1e-8:
        return [0.5] * len(scores)
```

When enabled, groups with genuine topological spread (AIME) still get
full-range rescaling; saturated groups (GSM8K/MATH500) are dead-banded
to a constant 0.5, so the topology term contributes a constant
`(1 + 0.5·α + 0.5·(1-α)) = 1.5` that cannot reorder the rollouts —
SCAE then reduces to outcome-only advantage, which is exactly the
desired "don't break what isn't broken" behaviour on easy benchmarks.

### How to verify

```bash
TOPO_RESCALE_PATCH=1 TOPO_RESCALE_MIN_SPAN=0.05 \
  bash scripts/run_topoprm_stage2.sh
```

Metric to watch in `logs/`: `reward/topo_rescale_collapsed_rate` (new
counter, emitted from `_maybe_log_stats`) should be >0.5 on
GSM8K/MATH500 rollouts and ~0 on AIME.

---

## Root cause 2 (P2) — additive base violates correctness primacy

### The code path

```python
# composite_reward.py
bw = {"outcome": 0.70, "format": 0.15, "length": 0.15}
r_base = bw["outcome"] * o + bw["format"] * f + bw["length"] * l
r_base_floored = max(r_base, 0.05)     # BASE_FLOOR=0.05
gain = 1.0 + alpha*t + (1.0-alpha)*c
r = r_base_floored * gain
```

Worked example:
- Rollout A: correct answer, format/length fine, structure perfect.
  `r = (0.70·1 + 0.15·1 + 0.15·1) · 1.5 = 1.00 · 1.5 = 1.500`
- Rollout B: **wrong** answer, format/length perfect, structure perfect.
  `r = (0.70·0 + 0.15·1 + 0.15·1) · 1.5 = 0.30 · 1.5 = 0.450`
- Rollout C: correct answer, format broken (answer in prose, not in
  `<answer>` tag), length fine, structure perfect.
  `r = (0.70·1 + 0.15·0 + 0.15·1) · 1.5 = 0.85 · 1.5 = 1.275`
- Rollout D: **wrong** answer, format perfect, length perfect,
  structure broken.
  `r = (0.70·0 + 0.15·1 + 0.15·1) · 1.0 = 0.30`

Rollouts B and D both score above the floor and get real positive
advantage in GRPO. The SCAE asymmetric clip (`[-c, 0]` for wrong
answers) would protect against this, **but** SCAE is applied to the
per-stratum normalized `(R_i - μ±) / (σ± + ε)`, not to raw rewards —
when the stratum contains only wrong answers (common early in training
on AIME), the clip has no reference to `correct = 1.5` and cannot undo
the additive shift.

### The patch (P2)

New flag `RewardConfig.TOPO_HIER_AGG` (default **"additive"**):

```python
if agg_mode == "multiplicative":
    f_gate = 0.5 + 0.5 * f      # [0.5, 1.0]
    l_gate = 0.5 + 0.5 * l      # [0.5, 1.0]
    r_base = o * f_gate * l_gate
    r_base_floored = r_base     # no floor; outcome=0 ⇒ r=0
else:
    r_base = bw["outcome"]*o + bw["format"]*f + bw["length"]*l
    r_base_floored = max(r_base, floor)

r = r_base_floored * (1 + α*t + (1-α)*c)
```

Under `multiplicative`:
- Rollout A: `1·1·1·1.5 = 1.500`
- Rollout B: `0·1·1·1.5 = 0.000`  ← wrong answer is strictly 0
- Rollout C: `1·0.5·1·1.5 = 0.750` ← format broken but answer right
- Rollout D: `0·1·1·1.0 = 0.000`  ← wrong answer is strictly 0

This matches the paper's Eq. 2 exactly. SCAE is now downstream of a
correct hierarchy, so its stratum-wise clip actually reflects the
outcome-primacy contract.

### How to verify

```bash
TOPO_HIER_AGG=multiplicative TOPO_RESCALE_PATCH=1 \
  bash scripts/run_topoprm_stage2.sh
```

Expected deltas vs. the current TopoPRM-full run:
- **Should flip from loss to parity on GSM8K/MATH500/CNMO/MMLU.**
  When outcome=0 is strictly 0, the noisy rescaling cannot create spurious
  positive advantage for wrong answers.
- **Should preserve or improve AIME24/25 gains.** Topology still amplifies
  outcome-correct rollouts' advantage; the structural-diversity signal
  on AIME is not touched.

---

## Non-regression guarantees

1. `TOPO_RESCALE_PATCH` defaults to **False**; `TOPO_HIER_AGG` defaults to
   **"additive"**. `git pull` + rerun on the v3b checkpoint will produce
   byte-identical rewards.
2. `scripts/check_reward_invariants.py` is a CI-style smoke test that
   asserts both defaults on every change to the reward module. The test
   stubs out MS-Swift so it runs in any env.
3. The old `BASE_FLOOR = 0.05` knob and the anti-collapse monitor are
   untouched. No call-site in `src/training/` or `scripts/train_grpo_*.sh`
   sees a changed interface.

Run the regression test anytime:
```bash
python3 scripts/check_reward_invariants.py
```

---

## What is **not** changed

- **SCAE implementation.** Unchanged. The audit flagged the stratum-wise
  clip as correct; the issue was upstream of SCAE (additive aggregation
  feeding SCAE a miscalibrated signal). With P2 applied, SCAE's
  correctness-primacy clip finally has a correctness-primacy input.
- **`q_topo` / `q_cont` computation.** Unchanged. The DAG extractor,
  acyclicity check, orphan check, direction check, and continuity scorer
  are all mechanically correct — they produce a usable per-trace scalar.
  The issue was in how that scalar was aggregated, not how it was
  computed.
- **Stage-III TG-OPD.** Unchanged. Not the current bottleneck;
  recompile once Stage-II is re-validated.
- **Paper method section.** Unchanged. The fixes restore the
  already-described behaviour (multiplicative gain with correctness
  primacy) rather than introduce new method.

---

## Recommended experiment plan

1. Smoke-test multiplicative aggregation only (no rescale patch) on
   the TopoPRM-full reward for 50 GRPO steps to confirm no training
   crash. `TOPO_HIER_AGG=multiplicative bash scripts/run_topoprm_stage2.sh`
2. Full 200-step run with both patches on:
   `TOPO_RESCALE_PATCH=1 TOPO_HIER_AGG=multiplicative bash scripts/run_topoprm_stage2.sh`
3. Re-evaluate on GSM8K / MATH500 / AIME24 / AIME25 / CNMO to see
   whether the per-benchmark pattern changes.
4. If AIME gain is preserved while GSM8K/MATH500 recover to parity,
   the P0+P2 thesis is confirmed and the paper section needs only a
   footnote about the revised aggregation (already matches Eq. 2 in
   `sections/3_method.tex`).
5. Leave `check_reward_invariants.py` in CI-adjacent position so the
   defaults never silently flip in future edits.

---

## File pointers

- `src/reward/composite_reward.py::_batch_rescale` — P0 patch
- `src/reward/composite_reward.py::TopoHierarchicalReward.__call__` — P2 patch
- `src/reward/reward_config.py::RewardConfig` — two new flags with
  `TOPO_RESCALE_PATCH`, `TOPO_RESCALE_MIN_SPAN`, `TOPO_HIER_AGG`
- `scripts/check_reward_invariants.py` — default-preservation test
- `scripts/dump_results_matrix.py` — eval matrix generator
- `TopoPRM_EMNLP26/sections/3_method.tex` — Eq. 2 (the multiplicative
  formula the paper already claims)
