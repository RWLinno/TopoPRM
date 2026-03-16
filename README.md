# TopoPRM

**Topology-aware Process Reward Model for Structural Math Reasoning via Reinforcement Fine-Tuning**

TopoPRM combines topological structure rewards and continuity rewards as verifiable process-level signals for RLVR, enabling structurally coherent and compact mathematical reasoning chains. It further applies reverse-KL distillation to compress long reasoning traces into concise student models.

## Quick Start

```bash
# 1. Install
cd /mnt/users/rwl/topoprm
conda create -n topo python==3.12
conda activate topoprm
pip install -r requirements.txt

# 2. Data pipeline (parse → clean → DAG → SFT → GRPO)
export PYTHONPATH=$(pwd):${PYTHONPATH:-}
bash scripts/run_data_pipeline.sh

# 3. SFT training
bash scripts/run_sft.sh

# 4. GRPO training (with topo+continuity reward)
bash scripts/run_grpo.sh
```

## Project Structure

```
topoprm/
├── src/
│   ├── dag/           # DAG datastructures (Node, ReasoningDAG, compress)
│   ├── data/          # Data pipeline (parse, clean, build_dag, prepare_sft/grpo)
│   ├── reward/        # Reward functions (outcome, format, topo, continuity, composite)
│   └── eval/          # Evaluation (benchmark_runner, critique_eval)
├── configs/           # ms-swift training configs
├── experiments/       # NeurIPS experiment configs & launch scripts
├── scripts/           # Shell scripts for data & training
├── docs/              # Design documents
└── tests/             # Unit tests (86 tests)
```

## Reward Design

| Component  | Weight | Description |
|------------|--------|-------------|
| Outcome    | 0.40   | Score accuracy (exact/close match) |
| Format     | 0.15   | `<think>/<answer>` tags + valid JSON |
| Topology   | 0.20   | DAG acyclicity, no orphan conclusions, direction consistency |
| Continuity | 0.15   | Step-by-step expression/claim traceability |
| Length     | 0.10   | Penalise verbose outputs (>2000 chars) |

## Experiment Configs

See `experiments/` for full NeurIPS experiment configurations:

```bash
# Main experiment: SFT → GRPO (topo+continuity)
bash experiments/run_main.sh

# Ablation: outcome-only reward
bash experiments/run_ablation_outcome_only.sh

# Ablation: no topo reward
bash experiments/run_ablation_no_topo.sh

# Distillation: reverse-KL compression
bash experiments/run_distill.sh
```
