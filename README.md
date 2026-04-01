# TopoPRM: Deterministic Verifiable Process Rewards + Reverse-KL Distillation

> **NeurIPS 2026 Submission** | Qwen3-32B | MS-Swift GRPO | Process-Aware Distillation

## Overview

TopoPRM is a two-contribution framework for mathematical reasoning post-training:

1. **Verifiable Process Reward Model (deterministic PRM)**
   - Converts free-form reasoning traces into an inferred dependency DAG
   - Computes deterministic **topology reward** (global dependency regularity)
   - Computes deterministic **continuity reward** (local support traceability)
   - Optional **GAT-based topology scorer** with Laplacian position encoding
   - Rewards are reproducible and auditable once a trace is generated

2. **Reverse-KL reasoning distillation**
   - Uses a graph-enriched teacher trained with sparse but verifiable rewards
   - Applies process-aware trace filtering before distillation
   - Compresses teacher reasoning behavior into compact chain-like student reasoning

Important scope note: this project provides **verifiable structural/process rewards**, not full semantic proof verification.

## Key Results

| Metric | TopoPRM (GRPO) | Outcome-Only | Improvement |
|--------|---------------|-------------|-------------|
| Overall Critique Acc | 29.7% | 16.3% | +13.4 |
| Format Compliance | 94.6% | 89.0% | +5.6 |
| Avg Response Length | 364 tok | 410 tok | -11.2% |
| Distilled 8B GSM8K | 82.5% | — | — |
| Distilled 8B MATH-500 | 59.8% | — | — |

## Quick Start

### Installation

```bash
conda create -n topoprm python=3.12 -y
conda activate topoprm
pip install -r requirements.txt
```

### Recommended Repro Order

```bash
# 0) env
conda activate topoprm

# 1) benchmark collection
bash scripts/download_benchmarks.sh

# 2) data pipeline
bash scripts/run_data_pipeline.sh

# 3) SFT
bash scripts/run_sft.sh

# 4) GRPO with hierarchical reward (recommended)
bash scripts/run_grpo.sh grpo_hierarchical

# 4b) Ablation variants
TOPO_ABLATION_CONFIG=configs/ablation_no_topo.yaml bash scripts/run_grpo.sh grpo_hierarchical
TOPO_ABLATION_CONFIG=configs/ablation_no_continuity.yaml bash scripts/run_grpo.sh grpo_hierarchical

# 5) evaluation
bash scripts/run_eval_all.sh
python3 -m src.eval.export_paper_tables --eval_dir output/eval --output output/eval/paper_table_summary.csv

# 6) distillation (optional)
bash scripts/generate_distill_data.sh

# 7) cleanup (optional, dry-run first)
bash scripts/cleanup_experiments.sh
```

## Reward Design

### Aggregation Strategies

TopoPRM supports multiple reward aggregation strategies, controlled via YAML config:

| Strategy | Key | Description |
|----------|-----|-------------|
| **Hierarchical** | `topo_hierarchical` | `R_base * (1 + α*topo + (1-α)*cont)` with anti-collapse |
| Linear | `topo_composite` | Weighted sum with dynamic adaptation |
| Multiplicative Gate | `topo_composite_mulgate` | `R_base * (1 + α*topo + β*cont)` |
| Confidence Gate | `topo_composite_confgate` | Outcome-confidence gated process rewards |
| SCAE | `topo_composite_scae` | Correctness-first stratified shaping |

### Ablation Configuration

All ablation switches are controlled via YAML:

```yaml
# configs/ablation_template.yaml
ablation:
  reward_aggregation: hierarchical
  topo_scorer: rule_based    # rule_based | gat | hybrid
  use_topo_reward: true
  use_continuity_reward: true
  alpha: 0.60
  reward_noise_eps: 0.01
  min_reward_std: 0.005
```

Set `TOPO_ABLATION_CONFIG` env var to point to your config file.

### Topology Scorer Options

- **rule_based** (default): Deterministic DAG quality metrics (acyclicity, orphan ratio, etc.)
- **gat**: Lightweight GAT (2-layer, 64-dim, 4 heads) with Laplacian position encoding
- **hybrid**: Average of rule-based and GAT scores

## Distillation Design

- Teacher traces filtered by process-aware quality criteria (`R_total > 0.7`)
- Student trained by reverse-KL objective on filtered traces
- Goal: preserve quality while reducing reasoning cost and verbosity

## Project Structure

```text
TopoPRM/
├── configs/
│   ├── grpo_hierarchical.yaml    # Main GRPO config (recommended)
│   ├── ablation_template.yaml    # Ablation switch template
│   ├── ablation_no_topo.yaml     # Ablation: no topology reward
│   ├── ablation_no_continuity.yaml
│   └── gat_topo.yaml             # GAT scorer config
├── scripts/
│   ├── cleanup_experiments.sh    # Conservative experiment cleanup
│   └── ...
├── src/
│   ├── dag/                      # DAG extraction and compression
│   ├── data/                     # Data pipeline
│   ├── distill/                  # Reverse-KL distillation
│   ├── eval/                     # Evaluation
│   ├── prm/                      # PRM interface (re-exports)
│   └── reward/
│       ├── composite_reward.py   # All aggregation strategies
│       ├── topo_reward.py        # Rule-based topology reward
│       ├── gat_topo_reward.py    # GAT topology scorer
│       ├── topo_position_encoding.py  # Laplacian PE
│       ├── continuity_reward.py
│       ├── format_reward.py
│       └── outcome_reward.py
├── paper/                        # NeurIPS 2026 LaTeX source
├── docs/
└── output/
```

## Citation

```bibtex
@article{topoprm2026,
  title={Deterministic Verifiable Process Rewards and Reverse-KL Distillation for Mathematical Reasoning},
  author={Ruan, Weilin},
  year={2026}
}
```

## License

This project is for research purposes only.
