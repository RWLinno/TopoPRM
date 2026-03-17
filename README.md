# TopoPRM: Topology-Aware Process Reward for Verifiable Mathematical Reasoning

> **NeurIPS 2025 Submission** | Qwen3-32B | MS-Swift GRPO | Verifiable Rewards

## Overview

TopoPRM is a **topology-aware process reward framework** for reinforcement fine-tuning of mathematical reasoning models. Unlike outcome-only RLVR methods that reward only correct final answers, TopoPRM evaluates the *structural quality* of the reasoning process by extracting and analyzing the implicit directed acyclic graph (DAG) of inter-step dependencies.

**Key innovations:**
- **Topological Structure Reward** — Extracts the reasoning DAG and evaluates acyclicity, connectivity, and directional consistency using deterministic algorithms. No learned reward model needed.
- **Continuity Reward** — Verifies that each reasoning step references expressions or conclusions from prior steps, penalizing unsupported logical jumps.
- **Stratified Clipping Advantage Estimation (SCAE)** — Prevents reward hacking by hierarchically prioritizing answer correctness over auxiliary process rewards during GRPO optimization.
- **Reverse-KL Distillation** — Compresses the improved reasoning behavior from a 32B teacher into smaller student models.

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    TopoPRM Pipeline                       │
├──────────────┬───────────────────────────────────────────┤
│  Stage 1     │  SFT on structured math reasoning data    │
│  Stage 2     │  GRPO with TopoPRM composite reward       │
│              │  ├─ Outcome Reward (answer correctness)   │
│              │  ├─ Topological Structure Reward (DAG)    │
│              │  ├─ Continuity Reward (step coherence)    │
│              │  ├─ Format Reward (output structure)      │
│              │  └─ Length Reward (conciseness)            │
│  Stage 3     │  Reverse-KL distillation to 14B/7B       │
└──────────────┴───────────────────────────────────────────┘
```

## Quick Start

### Prerequisites

```bash
# Activate the swift environment
conda activate swift  # or: export PATH="/path/to/conda_env/swift/bin:$PATH"

# Verify ms-swift is available
swift --version
```

### 1. SFT Training

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NPROC_PER_NODE=8
swift sft --config configs/sft_qwen3_32b.yaml
```

### 2. GRPO Training (with vLLM acceleration)

```bash
# Option A: Colocate mode (single command)
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NPROC_PER_NODE=8
swift rlhf --rlhf_type grpo --config experiments/configs/grpo_main.yaml

# Option B: Server mode (recommended for large models)
# Terminal 1: Launch vLLM rollout server
bash scripts/start_rollout_server.sh

# Terminal 2: Launch GRPO training
bash scripts/run_grpo.sh grpo_main
```

### 3. Ablation Experiments

```bash
bash scripts/run_grpo.sh grpo_outcome_only    # outcome reward only
bash scripts/run_grpo.sh grpo_no_topo          # remove topology reward
bash scripts/run_grpo.sh grpo_no_continuity    # remove continuity reward
```

### 4. Evaluation

```bash
bash scripts/run_eval.sh <adapter_checkpoint_path> <experiment_name>
# Example:
bash scripts/run_eval.sh output/grpo_main/v1/checkpoint-150 grpo_main
```

## Project Structure

```
topoprm/
├── src/
│   ├── dag/                    # DAG data structures and algorithms
│   ├── data/                   # Data pipeline (parsing, cleaning, DAG building)
│   ├── reward/                 # Reward functions
│   │   ├── composite_reward.py # Full TopoPRM composite reward (5 components)
│   │   ├── ablation_rewards.py # Ablation variants
│   │   ├── outcome_reward.py   # Answer correctness checking
│   │   ├── topo_reward.py      # DAG structural evaluation
│   │   ├── continuity_reward.py# Step-to-step coherence
│   │   └── format_reward.py    # Output format compliance
│   └── eval/                   # Evaluation (critique quality, benchmarks)
├── configs/                    # SFT configurations
├── experiments/configs/        # GRPO experiment configurations
├── scripts/                    # Training, evaluation, and utility scripts
├── paper/                      # NeurIPS paper (LaTeX)
├── data/                       # Training and test data
│   ├── sft_ready/              # SFT training data (5,000 samples)
│   ├── grpo_ready/             # GRPO prompts (637 samples)
│   └── test/                   # Test sets (7,595 samples)
└── docs/                       # Documentation and progress tracking
```

## Training Configuration

| Parameter | SFT | GRPO |
|-----------|-----|------|
| Base Model | Qwen3-32B | Qwen3-32B + SFT adapter |
| Method | LoRA (rank=64, alpha=128) | LoRA (rank=64, alpha=128) |
| Data | 5,000 samples | 637 critique prompts |
| Generations | - | 8 per prompt |
| Learning Rate | 5e-5 | 5e-6 |
| Epochs | 3 | 1 |
| Framework | MS-Swift 4.x | MS-Swift 4.x + vLLM |

## Citation

```bibtex
@article{topoprm2026,
  title={TopoPRM: Topology-Aware Process Reward for Verifiable Mathematical Reasoning},
  author={Ruan, Weilin},
  year={2026}
}
```

## License

This project is for research purposes only.
