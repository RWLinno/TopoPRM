# TopoPRM Handoff Prompt for New Server / Session

> Last updated: **2026-05-08**. Use this document as a self-contained prompt when starting a new Cursor agent session to continue the project.

## Quick Context for the Agent

You are continuing work on **TopoPRM**, a topology-aware process reward framework for mathematical reasoning. The project targets **EMNLP 2026 (ARR May 25 deadline)**.

**Core idea**: Sequential reasoning traces are topologically modeled as DAGs (not chains). DAG structural properties provide deterministic, training-free process rewards for GRPO. The trained model produces more structured reasoning that can be compressed via topology-verified self-distillation (TVSD).

**Pipeline**: Data Processing (text ? DAG) ? Reward Aggregation (DAG ? hierarchical reward) ? Post-Training (SFT ? GRPO ? TVSD)

## Step 1: Environment Setup

```bash
cd /path/to/TopoPRM
git checkout exp_May8
bash setup.sh
# Or manually:
conda create -n topoprm python=3.12 -y && conda activate topoprm
pip install -r requirements.txt
python -c "from src.reward.topo_reward import TopoReward; print('OK')"
```

## Step 2: Download Data and Checkpoints

```bash
# Data (DAG training data + eval results)
huggingface-cli download rwlinno/topoprm-data --repo-type dataset --local-dir data/hf_download
cp data/hf_download/train_public.jsonl data/grpo_ready/

# Checkpoints (SFT adapter)
huggingface-cli download rwlinno/topoprm-ckpts --local-dir output/hf_ckpts
```

If HF download fails, rebuild DAGs from scratch:
```bash
python3 scripts/build_dag_public.py --datasets gsm8k math --output_dir data/dag_public
```

## Step 3: Current Experiment Status

| Phase | Status | Artifacts |
|-------|--------|-----------|
| Phase 1: DAG Construction | DONE | `data/dag_public/` (19,472 DAGs), `data/grpo_ready/train_public.jsonl` |
| Phase 2: DR1-7B Baseline | DONE | `output/eval/baseline_dr1_7b_chat_*_metrics.json` (5 benchmarks) |
| Phase 3a: SFT | DONE | `output/sft_deepseek_r1_7b/final/` (LoRA adapter) |
| Phase 3b: GRPO | **NOT STARTED** | Next step |
| Phase 4: Eval with adapter | NOT STARTED | After Phase 3b |
| Phase 5: Paper sync | IN PROGRESS | Tables have ~estimates, need real data |
| AIME2026 + FrontierMath | DONE | `data/benchmarks/{AIME2026,FrontierMath}/` |
| PRM/DAG Quality Validation | DONE | `output/analysis/prm_dag_quality/` |

### Phase 2 Baseline Results (DeepSeek-R1-Distill-7B, chat template)

| Benchmark | pass@1 | pass@5 | maj@5 | prm@5 | avg_tok |
|-----------|--------|--------|-------|-------|---------|
| GSM8K | 83.5% | 90.8% | 82.6% | 83.5% | 599 |
| MATH-500 | 55.6% | 62.8% | 58.8% | 55.6% | 3412 |
| AIME 2024 | 23.3% | 33.3% | 30.0% | 23.3% | 4096 |
| CNMO 2024 | 23.3% | 40.0% | 30.0% | 23.3% | 4094 |
| MMLU | 42.5% | -- | -- | -- | 511 |

**Known issues**:
- MATH-500 accuracy is underestimated: `extract_number` cannot handle LaTeX answers like `\frac{14}{3}`. Fix: implement sympy-based comparison or use a reference evaluation library.
- AIME/CNMO hit `max_new_tokens=4096` truncation. Fix: increase to 8192+.
- prm@5 = pass@1 for baseline is expected (no TopoPRM training yet, PRM scores are trivial).

## Step 4: Experiment Plan (Priority Order)

### 4.1 Fix Answer Extraction (HIGH PRIORITY)
File: `scripts/bench_transformers.py`, functions `extract_number` and `answers_match_numeric`.
- Add LaTeX normalization (sympy or reference impl from MATH evaluation)
- Support `\frac{}{}`, `\left(...\right)`, `\text{}`, coordinate tuples
- Re-run Phase 2 baselines after fix

### 4.2 Run GRPO Training (Phase 3b)
```bash
CUDA_VISIBLE_DEVICES=0,1 python3 scripts/train_grpo.py \
  --sft_adapter output/sft_deepseek_r1_7b/final
```
Config: 200 steps, 4 generations/prompt, lr 5e-6, beta 0.04, TopoPRM hierarchical reward.
Output: `output/grpo_topoprm_deepseek_r1_7b/final/`

### 4.3 Run Phase 4 Evaluation
```bash
python3 scripts/bench_transformers.py \
  --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
  --adapter output/grpo_topoprm_deepseek_r1_7b/final \
  --label topoprm_dr1_7b \
  --benchmarks gsm8k math500 aime2024 cnmo2024 mmlu \
  --use_chat_template --sft_style \
  --num_samples_per_item 5 --k_values 1 5 \
  --max_new_tokens 8192
```

### 4.4 Run Ablation Variants
Same as 4.3 but with different reward configs:
- outcome-only: modify `train_grpo.py` to use `OutcomeOnlyReward`
- w/o topology: use `NoContinuityReward`
- w/o continuity: use `NoTopoReward`

### 4.5 TVSD Compression (Phase 5)
After GRPO produces a strong teacher:
1. Self-refinement (Phase III-A): `scripts/rollout_srt.py`
2. On-policy distillation (Phase III-B): requires student model (e.g., Qwen3.5-4B)

### 4.6 Fill Paper Tables
Replace all `~XX.X` estimates in `topoprm_paper/tables/public_results.tex` with real numbers.
Update `4_experiments.tex` narrative to match.

## Step 5: Paper Status

LaTeX source: `topoprm_paper/`
- `main.tex`: Currently NeurIPS style, **needs switch to ACL/EMNLP template before submission**
- Abstract, intro, method: Well-written, aligned with code
- Related work: Updated with ThinkPRM, GenPRM, RLKD, OPSDC, ExOPD
- Experiments: Setup aligned with trl-based scripts; public results have ~estimates
- Tables: `public_results.tex` has 12-column format (p@1/m@5/prm@5 x 3 benchmarks + MMLU)

## Key Files

| File | Purpose |
|------|---------|
| `src/dag/graph.py` | ReasoningDAG class |
| `src/dag/compress.py` | DAG compression (layering, contraction) |
| `src/data/build_dag.py` | Text ? DAG extraction |
| `src/reward/composite_reward.py` | All reward classes (hierarchical, gated, SCAE) |
| `src/reward/topo_reward.py` | Topology reward computation |
| `src/reward/continuity_reward.py` | Continuity reward |
| `scripts/train_sft.py` | SFT training (trl) |
| `scripts/train_grpo.py` | GRPO training with TopoPRM (trl) |
| `scripts/bench_transformers.py` | Unified evaluation script |
| `scripts/validate_prm_dag_quality.py` | PRM + DAG quality analysis |

## HuggingFace Repos

- Checkpoints: https://huggingface.co/rwlinno/topoprm-ckpts
- Data: https://huggingface.co/datasets/rwlinno/topoprm-data

## GitHub

- Repo: https://github.com/RWLinno/TopoPRM
- Branch: `exp_May8`
