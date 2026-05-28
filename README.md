# TopoPRM: Topology-Aware Process Reward Model for Mathematical Reasoning

<p align="center">
  <img src="docs/assets/topoprm_overview.png" width="80%" alt="TopoPRM Framework Overview">
</p>

> **TopoPRM** extracts dependency DAGs from reasoning traces and uses graph-structural signals to design hierarchical reward functions for GRPO post-training. The same DAG interface unifies data construction, reward design, and distillation.

## Highlights

- 🔬 **Topology-aware reward**: Evaluates reasoning quality via DAG structure (acyclicity, direction, step-alignment)
- 🎯 **Correctness-first aggregation**: Multiplicative design ensures incorrect answers can never be rewarded
- 📊 **ACE (Advantage Clipping Estimation)**: Stratified clipping preserves outcome primacy across strata
- 🚀 **Three-stage pipeline**: SFT → GRPO+TopoPRM → Topology-Guided On-Policy Distillation

## Results

### Qwen2.5-7B + TopoPRM

| Benchmark | pass@1 | pass@5 |
|-----------|--------|--------|
| MATH-500 | 66.8% | 72.6% |
| AIME'24 | 13.3% | 13.3% |
| AIME'25 | 10.0% | 26.7% |
| CNMO'24 | 47.0% | — |

### Qwen3.5-9B + TopoPRM

| Benchmark | pass@1 | pass@5 |
|-----------|--------|--------|
| AIME'24 | 3.3% | 6.7% |
| AIME'25 | 3.3% | 13.3% |
| Omni-MATH | 55.2% | 76.2% |
| OlympiadBench | ~42% | (running) |

### DeepSeek-R1-7B + TopoPRM

| Benchmark | pass@1 |
|-----------|--------|
| MATH-500 | 66.0% |
| OlympiadBench | 58.0% |
| MMLU | 84.0% |
| GPQA-Diamond | 36.0% |

## Quick Start

### Installation

```bash
pip install -r requirements.txt
# Requires: torch>=2.1, transformers, peft, ms-swift, math-verify, networkx
```

### Evaluation

```bash
# Evaluate a TopoPRM checkpoint
CUDA_VISIBLE_DEVICES=0 python scripts/bench_transformers.py \
  --model /path/to/Qwen3.5-9B \
  --adapter output/hf_ckpts/grpo_hier_9b_ckpt79 \
  --label topoprm_9b \
  --benchmarks math500 aime2024 olympiadbench \
  --use_chat_template \
  --num_samples_per_item 5 \
  --k_values 1 5 \
  --max_new_tokens 4096
```

### Training (GRPO + TopoPRM)

```bash
# Set environment variables
export TOPO_HIER_AGG=multiplicative
export TOPO_SCAE_PRESERVE_OUTCOME=1
export TOPO_RESCALE_PATCH=1
export TOPO_DAG_SENTENCE_FALLBACK=1
export TOPO_LENGTH_UNIT=tokens

# Launch training with ms-swift
swift rlhf configs/grpo_9b_from_ckpt79.yaml
```

### Running Tests

```bash
pytest tests/ -v
```

## Project Structure

```
TopoPRM/
├── src/
│   ├── reward/                    # Core reward functions
│   │   ├── composite_reward.py    # TopoHierarchical, TopoSCAE, TopoComposite
│   │   ├── outcome_reward.py      # Math answer verification (math_verify)
│   │   ├── format_reward.py       # Format compliance check
│   │   ├── topo_reward.py         # DAG structural quality scoring
│   │   ├── continuity_reward.py   # Reasoning chain continuity
│   │   └── reward_config.py       # Environment variable configuration
│   ├── data/                      # Data processing & DAG extraction
│   └── eval/                      # Evaluation utilities
├── configs/                       # Training YAML configs
├── scripts/                       # Evaluation & training scripts
│   ├── bench_transformers.py      # Unified benchmark runner
│   └── unified_eval_orchestrator.py  # Multi-GPU eval scheduler
├── data/grpo_ready/               # Training data (query + solution + reference_dag)
├── output/hf_ckpts/               # Pre-trained checkpoints
├── results/                       # Evaluation results (metrics JSON)
├── tests/                         # Unit tests
├── tutorials/                     # Visualization & analysis tools
└── docs/                          # Documentation
```

## Checkpoints (HuggingFace)

Available at [`rwlinno/topoprm-ckpts`](https://huggingface.co/rwlinno/topoprm-ckpts):

| Checkpoint | Base Model | Training | Steps |
|-----------|-----------|----------|-------|
| `grpo_hier_9b_ckpt79` | Qwen3.5-9B | GRPO + topo_hierarchical | 79 |
| `grpo_9b_from_ckpt79_ckpt120` | Qwen3.5-9B | TopoPRM v2 (from ckpt79) | 120 |
| `grpo_topoprm_dr1_7b_ckpt949` | DeepSeek-R1-Distill-Qwen-7B | GRPO + SCAE | 949 |
| `opd_qwen25_7b_stage3_ckpt200` | Qwen2.5-7B-Instruct | Stage 3 OPD | 200 |

## Method Overview

### Hierarchical Reward Aggregation

```
r = outcome × format_gate × length_gate × (1 + α·topo + (1-α)·continuity)
```

- **Outcome** (0 or 1): Binary correctness via math_verify
- **Format gate** (0.5~1.0): Checks `<think>` + `\boxed{}` structure
- **Length gate** (0.5~1.0): Penalizes too-short or too-long responses
- **Topo gain**: DAG quality (valid, acyclic, no orphans, direction, step-alignment)
- **Continuity**: Step-to-step traceability of expressions

### ACE (Stratified Clipping)

Separates correct/incorrect strata, normalizes within each, then maps:
- Correct stratum → `[floor_pos, clip_hi]` = `[0.3, 1.5]`
- Incorrect stratum → `[clip_lo, -floor_neg]` = `[-1.5, -0.3]`

Guarantees: `min(correct_rewards) > max(incorrect_rewards)`

## Configuration

Key environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `TOPO_HIER_AGG` | `additive` | Aggregation mode: `additive` or `multiplicative` |
| `TOPO_SCAE_PRESERVE_OUTCOME` | `False` | Enable correctness-first stratified mapping |
| `TOPO_RESCALE_PATCH` | `False` | Batch rescale dead-zone patch |
| `TOPO_DAG_SENTENCE_FALLBACK` | `False` | Sentence-level DAG extraction fallback |
| `TOPO_LENGTH_UNIT` | `chars` | Length measurement: `chars` or `tokens` |
| `TOPO_LENGTH_LOW` | `200` | Minimum acceptable length |
| `TOPO_LENGTH_HIGH` | `4096` | Maximum acceptable length |

## Citation

```bibtex
@article{ruan2026topoprm,
  title={TopoPRM: Topology-Aware Process Reward Model for Mathematical Reasoning},
  author={Ruan, Weilin},
  year={2026}
}
```

## License

MIT
