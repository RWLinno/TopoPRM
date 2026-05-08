# TopoPRM Session Summary (2026-04-07)

> This document records all work done in the 2026-04-07 conversation window, including experiment results, code improvements, new experiments launched, problem diagnosis and solutions.

---

## 1. Completed Experiment Results

### 1.1 Cross-Scale Ablation Full Eval Results

All GPU 0-3 experiments completed. Results sorted by Mid Acc descending:

| Experiment | Mid Acc | Mid Fmt | High Acc | High Fmt | Overall |
|---|---|---|---|---|---|
| **distill_rkl_8b_compact** | **60.0** | 94.0 | **58.0** | 97.0 | **59.0** |
| SFT 32B | 39.0 | 48.7 | 26.5 | 45.8 | 32.8 |
| GRPO 32B (full TopoPRM) | 37.2 | 94.6 | 22.4 | 87.1 | 29.8 |
| GRPO 9B no_topo | 34.0 | 85.0 | 28.0 | 82.0 | 31.0 |
| GRPO 9B hier ng4 | 33.0 | 86.5 | 29.0 | 81.5 | 31.0 |
| GRPO 32B (light200) | 33.0 | 93.0 | 25.5 | 89.0 | 29.3 |
| GRPO 9B hier mem70 | 31.5 | 86.0 | 26.5 | 85.5 | 29.0 |
| GRPO 9B outcome_only | 31.5 | 84.0 | 25.0 | 82.0 | 28.3 |
| GRPO 9B hier | 30.5 | 87.5 | 27.0 | 83.5 | 28.8 |
| GRPO 9B no_cont | 29.5 | 85.5 | 29.0 | 84.0 | 29.3 |
| SFT private_boost | 29.0 | 78.5 | 19.5 | 65.0 | 24.3 |
| GRPO 32B clipped | 29.0 | 93.0 | 21.5 | 88.5 | 25.3 |
| GRPO 32B no_topo | 23.0 | 91.0 | 16.5 | 86.0 | 19.8 |
| GRPO 7B outcome_only | 22.0 | 19.5 | 8.0 | 15.0 | 15.0 |
| GRPO 7B no_topo | 21.0 | 26.0 | 5.0 | 19.5 | 13.0 |
| GRPO 32B outcome_only | 19.5 | 89.0 | 13.0 | 85.5 | 16.3 |
| Baseline 32B zero-shot | 16.0 | 64.5 | 7.0 | 43.5 | 11.5 |
| GRPO 7B hier | 14.0 | 38.0 | 4.5 | 23.0 | 9.3 |
| GRPO 7B no_cont | 12.0 | 47.5 | 8.0 | 52.0 | 10.0 |

### 1.2 New Experiments This Round (9B mcl=4096 ablation + SFT cold-start)

| GPU | Experiment | Steps | Reward | Reward Std | Loss |
|-----|-----------|-------|--------|-----------|------|
| GPU0 | 9B outcome_only mcl4096 | 79/79 | 0 | 0 | 0 |
| GPU1 | 9B gated mcl4096 | 79/79 | 0.0125 | 0 | 0 |
| GPU2 | **9B SFT cold-start** | 626/626 | -- | -- | 0.329 |
| GPU3 | **7B SFT boost (5ep)** | 1560/1560 | -- | -- | 0.085 |
| GPU4 | 9B hier mcl4096 | 79/79 | 0.020 | 0.008 | 0.003 |
| GPU5 | 9B no_topo mcl4096 | 79/79 | **0.069** | **0.025** | 0.027 |

### 1.3 New Checkpoints Generated

```
output/sft_qwen35_9b/v0-20260407-*/checkpoint-*          # 9B SFT (2 epochs)
output/sft_qwen25_7b_boost/v0-20260407-*/checkpoint-*    # 7B SFT boost (5 epochs)
output/grpo_hierarchical_qwen35_9b_mcl4096/              # 9B hier mcl4096
output/grpo_no_topo_qwen35_9b_mcl4096/                   # 9B no_topo mcl4096
output/grpo_outcome_only_qwen35_9b_mcl4096/              # 9B outcome mcl4096
output/grpo_gated_qwen35_9b_mcl4096/                     # 9B gated mcl4096
```

---

## 2. Key Observations & Analysis

### Obs 1: Topology reward effectiveness is scale-dependent

- **32B: topo helps** (full 33.0 > no_topo 23.0, delta=+10.0)
- **9B: topo neutral/harmful** (no_topo 34.0 > full 30.5, delta=-3.5)
- **7B: topo severely harmful** (outcome_only 22.0 > hier 14.0, delta=-8.0)

Analysis: Small models lack representation capacity to jointly optimize outcome accuracy and topology structure. The topo reward introduces an extra optimization objective that interferes with outcome learning. 32B has enough capacity for both.

### Obs 2: Hierarchical reward causes reward hacking at 7B

- hier 7B: fmt=38.0% but acc=14.0% (learned format, worse answers)
- outcome_only 7B: fmt=19.5% but acc=22.0% (bad format, better answers)
- Negative correlation between format compliance and accuracy = reward hacking

### Obs 3: Knowledge distillation >> GRPO

- distill_rkl_8b_compact: Mid=60.0, High=58.0 (8B model)
- Best GRPO 32B: Mid=37.2, High=22.4
- Distilled 8B beats GRPO 32B by 60%+. Teacher signal >> reward shaping.

### Obs 4: no_topo has strongest signal at 9B mcl=4096

This round's 9B mcl=4096 ablation:
- **no_topo: reward=0.069, std=0.025** -- strongest signal, effective gradients
- hier: reward=0.020, std=0.008 -- weak signal
- gated: reward=0.0125, std=0 -- format partial credit but zero variance (no gradient)
- outcome: reward=0, std=0 -- completely zero signal

Consistent with mcl=1024 conclusion: no_topo is best at 9B.

### Obs 5: 9B previously used 32B's SFT adapter (cross-model issue)

9B had no SFT checkpoint of its own. All previous 9B GRPO experiments used 32B's `checkpoint-120` as adapter. This is a serious experimental design flaw. Now fixed: 9B SFT cold-start completed (`checkpoint-626`).

---

## 3. Code Changes

### 3.1 New Reward: Lexicographic Gated Reward (`topo_gated`)

**Problem**: Original hierarchical reward has hardcoded weights (0.70/0.15/0.15), reviewers will question.

**New design**:
```
R = outcome + delta * format * (1 + epsilon * q)
```

Parameter derivation (not tuning):
- `delta = 0.1`: derived from OutcomeReward discretization (min_gap=0.167, delta < 0.167)
- `epsilon = 0.5`: influence bound (delta*(1+epsilon)=0.15 < 0.167)
- `q = topo_w * batch_rescale(topo) + (1-topo_w) * batch_rescale(continuity)`
- `TOPO_WEIGHT`: env var `TOPO_GATED_TOPO_W` (default 0.5)

Properties:
- Outcome primacy: correct answer always ranks above incorrect
- Zero arbitrary weights: delta derived from outcome structure
- Ranking-safe: max format+process contribution (0.15) < min outcome gap (0.167)
- 0/40040 ranking violations (Monte Carlo verified)

Code: `src/reward/composite_reward.py` -> `TopoGatedReward`

### 3.2 FormatReward v2 (progressive curriculum)

```
<think> + <answer>JSON</answer> -> 1.0
<answer>JSON</answer> only     -> 0.5
<think> only (truncated)       -> 0.1
Nothing                        -> 0.0
```

Code: `src/reward/format_reward.py`

### 3.3 Truncation detection fix

- **Old**: fixed char threshold `TRUNC_CHAR_THRESHOLD=1800` (breaks at mcl>1024)
- **New**: semantic detection -- check if `</answer>` tag is present

### 3.4 TOPO_WEIGHT env var

`TopoGatedReward` now supports `TOPO_GATED_TOPO_W` env var:
- `0.0` = continuity-only
- `0.5` = equal mix (default)
- `1.0` = topo-only

---

## 4. Problems Encountered & Solutions

### P1: GPU 0/1 driver-level zombie processes (128GB VRAM stuck)

**Symptom**: 6 processes (PID 106960 etc) dead (`/proc/{pid}` gone) but nvidia-smi still reports 128GB usage.

**Tried**: `kill -9` (no effect), `nvidia-smi --gpu-reset` (not supported in container), persistence mode reset (no effect).

**Resolution**: GPUs eventually self-released (likely NVIDIA driver GC mechanism).

### P2: Gated reward all-zero signal

**Symptom**: gated reward = 0 on all models.

**Root cause chain**:
1. 87.3% of SFT data is English math (no `<think>/<answer>` tags)
2. Model never learned format tags -> format=0 -> gated reward=0
3. Even with format partial credit (`<think>` -> 0.1), all generations in batch have identical reward -> reward_std=0 -> GRPO zero gradient

**Solution**: FormatReward v2 partial credit + semantic truncation detection + 9B/7B SFT cold-start.

### P3: 9B had no SFT checkpoint

**Symptom**: 9B GRPO used 32B's SFT adapter (cross-model), 9B never learned output format.

**Solution**: Created `configs/sft_qwen35_9b.yaml`, completed 9B SFT on GPU2 (2 epochs, checkpoint-626).

### P4: 7B SFT format compliance only 19-52%

**Symptom**: 7B SFT checkpoint-624 has low format compliance, GRPO gets no format signal.

**Solution**: Created `configs/sft_qwen25_7b_boost.yaml` (5 epochs, lr=1e-4), completed on GPU3 (checkpoint-1560, loss=0.085).

### P5: TRUNC_CHAR_THRESHOLD breaks at mcl=4096

**Symptom**: All generations > 1800 chars -> all marked truncated -> q=0.5 -> reward_std=0.

**Solution**: Changed to semantic detection (check `</answer>` tag presence).

### P6: `run_sft.sh` hardcodes config path

**Symptom**: Script only reads `configs/sft.yaml` (32B), can't pass different config.

**Solution**: Launch directly with `swift sft --config configs/sft_qwen35_9b.yaml`.

---

## 5. Config Files Created

```
configs/sft_qwen35_9b.yaml                        # 9B SFT (2 epochs)
configs/sft_qwen25_7b_boost.yaml                   # 7B SFT boost (5 epochs, lr=1e-4)
configs/sft_qwen25_7b_v2.yaml                      # 7B SFT v2 (3 epochs, lr=1e-4)
configs/grpo_gated_qwen3_32b.yaml                  # 32B gated GRPO
configs/grpo_gated_qwen35_9b_mcl4096.yaml          # 9B gated mcl=4096
configs/grpo_gated_qwen35_9b_ng4_mcl4096.yaml      # 9B gated ng4+mcl=4096
configs/grpo_gated_qwen25_7b_mcl2048.yaml          # 7B gated mcl=2048
configs/grpo_outcome_only_qwen35_9b_mcl4096.yaml   # 9B outcome mcl=4096
configs/grpo_outcome_only_qwen25_7b_mcl2048.yaml   # 7B outcome mcl=2048
configs/grpo_hierarchical_qwen35_9b_mcl4096.yaml   # 9B hier mcl=4096
configs/grpo_no_topo_qwen35_9b_mcl4096.yaml        # 9B no_topo mcl=4096
```

---

## 6. Currently Running Experiments

| GPU | Experiment | Config | Status |
|---|---|---|---|
| GPU0 | 9B outcome_only mcl=4096 | `grpo_outcome_only_qwen35_9b_mcl4096` | Running |
| GPU1 | 9B gated mcl=4096 | `grpo_gated_qwen35_9b_mcl4096` | Running |
| GPU2 | 9B SFT cold-start | `sft_qwen35_9b` | Completed |
| GPU3 | 7B SFT boost | `sft_qwen25_7b_boost` | Completed |
| GPU4 | 9B hier mcl=4096 | `grpo_hierarchical_qwen35_9b_mcl4096` | Running |
| GPU5 | 9B no_topo mcl=4096 | `grpo_no_topo_qwen35_9b_mcl4096` | Running |

---

## 7. Next Steps

1. **Eval new checkpoints**: run `run_eval_light_private.sh` on all 6 new checkpoints, compare mcl=1024 vs mcl=4096
2. **9B gated GRPO with own SFT**: use `sft_qwen35_9b/checkpoint-*` instead of 32B adapter
3. **7B gated GRPO with boost SFT**: use `sft_qwen25_7b_boost/checkpoint-*` instead of old checkpoint-624
4. **Update LaTeX tables**: add mcl=4096 ablation results
5. **Run public benchmarks**: GSM8K, MATH-500, CMath etc., fill `public_results.tex`
6. **Scale-dependent analysis for paper**: topo reward effectiveness varies with model scale -- valuable contribution
7. **Explore distillation + GRPO combo**: distill first then GRPO may be the optimal path


---

## 8. Benchmark & Process Label Analysis

### Private benchmark (Chinese math critique)

Test data: `data/test/light_middle_200.jsonl`, `data/test/light_high_200.jsonl`

Fields with process-level labels:
- `step_results`: per-step correctness labels (e.g. `['correct', 'wrong', 'wrong']`)
- `analyse_str`: per-step error descriptions (e.g. `['discriminant calculation error']`)
- `user_step_split_emb`: step boundary encoding
- `std_score`: ground-truth total score

Eval metrics (from `src/eval/critique_eval.py`):
- `score_accuracy`: whether predicted score matches ground truth
- `format_compliance`: whether output follows `<answer>JSON</answer>` format
- `error_identification_f1`: F1 of identifying which steps are wrong
- `step_coverage`: fraction of steps addressed in critique

**Conclusion**: Private benchmark has rich process-level labels. Eval is well-supported.

### Public benchmarks (GSM8K, MATH-500, CMath, etc.)

Data in `data/benchmarks/`:
- GSM8K: 1319 test samples, keys = `[question, answer]`
- MATH-500: 500 test samples, keys = `[problem, solution, answer, subject, level]`
- CMath: 1098 test samples, keys = `[grade, question, golden, reasoning_step, num_digits]`

**No process-level labels** -- only final answers. Current `public_results.tex` is mostly TBD.

### Do we need LLM-as-judge for public benchmarks?

**Short answer: No, not for standard math benchmarks.**

For GSM8K/MATH-500/CMath, the standard evaluation is **answer extraction + exact match**:
1. Extract the final numerical/symbolic answer from model output
2. Compare with ground truth (exact match or numerical equivalence)

This is the standard protocol used by all papers (DeepSeek-R1, Qwen, etc.) and does not require LLM-as-judge.

**However**, if we want to evaluate **critique quality** (our actual task) on public data, we would need:
1. A student solution to critique (we'd need to generate wrong solutions first)
2. Process-level labels for the student solution (which steps are wrong)
3. An evaluator for the critique output

This is essentially a different task from standard math benchmarks. Our private benchmark already covers this. For the paper, the standard approach is:
- **Private benchmark**: evaluate critique quality (score_accuracy, error_identification_f1, etc.)
- **Public benchmark**: evaluate general math reasoning ability (GSM8K, MATH-500 pass@1)
- These serve different purposes: private shows task-specific performance, public shows the model hasn't lost general math ability

### Recommendation

1. Run standard pass@1 eval on GSM8K, MATH-500, CMath to fill `public_results.tex`
2. No LLM-as-judge needed for public benchmarks
3. The private benchmark with process labels is our main evaluation
4. If reviewers ask for process-level eval on public data, we can argue our private benchmark already covers this with 7,595 samples (2,181 middle + 5,414 high)

---

## 9. Currently Running (as of end of session)

| GPU | Task | Status | ETA |
|---|---|---|---|
| 0 | 9B gated GRPO (own SFT adapter) | Training | ~7h |
| 1 | 7B gated GRPO (boost SFT adapter) | Training | ~3h |
| 2 | Eval: 9B no_topo mcl4096 | Middle 8% | ~1.5h |
| 3 | Eval: 9B hier mcl4096 | Middle 8% | ~1.5h |
| 4 | Eval: 9B outcome mcl4096 | Middle 8% | ~1.5h |
| 5 | Eval: 9B gated mcl4096 | Middle 8% | ~1.5h |
| 6 | Eval: 9B SFT | Middle 3% | ~4h |
| 7 | Eval: 7B SFT boost | Middle 6% | ~2h |


---

## 8. Benchmark & Process Label Analysis

### Private benchmark (Chinese math critique)
- Data: `data/test/light_middle_200.jsonl` (200 middle school), `data/test/light_high_200.jsonl` (200 high school)
- Full: `data/test/infer_*_full_*.json` (2181 middle, 5414 high)
- **Has process labels**: `step_results` field (list of per-step correctness: correct/wrong), `analyse_str` (error descriptions)
- Eval script: `scripts/run_eval_light_private.sh <model> <adapter|none> <name>`
- Metrics: score_accuracy, format_compliance, error_identification_f1, step_coverage

### Public benchmarks (math problem solving)
- Downloaded to `data/benchmarks/`: GSM8K (1319 test), MATH-500 (500 test), AIME2024 (30), CMath, etc.
- **NO process labels** -- only final answer (e.g. GSM8K: `question`, `answer`; MATH-500: `problem`, `solution`, `answer`)
- Eval script: `scripts/run_public_benchmarks.sh <model> [adapter] [label]`
- Uses `swift eval` internally for standard benchmarks

### Process label gap on public benchmarks
- Our private benchmark has step-level labels (`step_results`, `analyse_str`) -- this is what makes TopoPRM evaluation meaningful
- Public benchmarks only have final answers -- we can only measure answer accuracy, not process quality
- **Options for public benchmark process evaluation**:
  1. LLM-as-Judge: use GPT-4o/Claude to annotate process quality on public benchmark outputs
  2. Use existing process-annotated datasets (PRM800K, MathShepherd) if they overlap with our benchmarks
  3. Focus paper narrative on private benchmark for process metrics, public for answer accuracy only

### API keys available in `.env`
- DashScope (Qwen API): for Chinese LLM-as-Judge
- OpenAI: for GPT-4o as judge
- Anthropic: for Claude as judge

---

## 9. Currently Running (as of session end)

### Evals (GPU 2-7, ~30min remaining)
- GPU2: eval 9B no_topo mcl4096 (76%)
- GPU3: eval 9B hier mcl4096 (84%)
- GPU4: eval 9B outcome mcl4096 (72%)
- GPU5: eval 9B gated mcl4096 (71%)
- GPU6: eval 9B SFT (23%)
- GPU7: eval 7B SFT boost (56%)

### GRPO training (GPU 0-1)
- GPU0: 9B gated + own SFT adapter (16/79, reward=0.013, **reward_std=0.0006 > 0** -- first time gated has non-zero variance!)
- GPU1: 7B gated + boost SFT adapter (105/318, reward=0 -- 7B boost SFT still insufficient)

### Key finding: 9B own SFT makes gated reward work
- With 32B's SFT adapter: gated reward_std=0 (zero gradient)
- With 9B's own SFT adapter: reward_std=0.0006 > 0 (non-zero gradient!)
- This confirms that proper per-model SFT cold-start is critical for gated reward

---

## 10. DAG闭环升级（2026-04-08）

### 10.1 提取层修复
- `extract_claims` 升级为句级语义提取，过滤不完整残句（如“所有有两种运输方案：”）。
- 新增 `extract_claim_keys` 专用于依赖匹配，避免显示用句级 claim 影响图连边。
- `parse_answer_to_dag_debug` 支持 `reference_dag` 输入；`local_verdict` 采用 hybrid 策略：
  1) 优先继承 reference verdict；
  2) 缺失时按规则回填 `correct/incorrect/unverifiable`。
- 顺序弱边从默认全量链改为 `adaptive` 模式（仅在缺少强依赖时补边），并输出边来源统计。

### 10.2 奖励层增强（公式逐项可审计）
- `TopoReward` 增加公式项诊断并逐样本保存：
  - `term_base = λ_b * I[|V|>0]`
  - `term_acyclic = λ_a * I[acyclic]`
  - `term_orphan = λ_o * I[rho_orphan=0]`
  - `term_delta = λ_d * delta`
  - `term_kappa = λ_k * kappa`
- 诊断包含 `rho_orphan/delta/kappa/lambda_*/term_*/denom/r_topo`，可直接对照论文公式审计。
- 无参考图时自动移除 `kappa` 项（`lambda_kappa=0`）并重算归一化分母。

### 10.3 压缩与可视化闭环
- 新增 `compress_dag_by_layers`：按拓扑层压缩 DAG（每层一个节点），生成层压缩链。
- GUI 改为单视图 `layer_boxes`：
  - 同图展示顺序边与依赖边；
  - 层级虚线框；
  - 箭头指向节点边缘；
  - 边按类型曲率分离避免重叠；
  - 显示 `Lk -> Ck` 层到压缩链映射；
  - 展示压缩率与层数统计，打通“拓扑奖励 -> 推理压缩”闭环。

### 10.4 测试与回归
- 新增/增强测试覆盖：
  - claim 残句过滤；
  - hybrid verdict 继承；
  - 非链式依赖边存在性；
  - `r_topo` 分项一致性（term 和总分一致）；
  - layer 压缩映射正确性。
- 关键回归：`tests/test_build_dag.py + tests/test_graph.py + TestTopoReward` 全部通过。
