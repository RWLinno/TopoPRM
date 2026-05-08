# TopoPRM: Topology-Aware Process Rewards for Verifiable Mathematical Reasoning

> **EMNLP 2026 (ARR May cycle)** | DeepSeek-R1-Distill-7B + Qwen3.5-9B | TRL GRPO | Deterministic DAG Rewards

## Overview

TopoPRM treats sequential reasoning traces as implicitly structured graphs. The framework:

1. **Topological Modeling**: Extracts dependency DAGs from chain-of-thought outputs via deterministic rule-based parsing (no LLM calls, no learned models)
2. **Reward Aggregation**: Computes topology reward (global DAG validity) and continuity reward (local step traceability), combined with outcome/format/length via hierarchical multiplicative aggregation
3. **Post-Training Optimization**: SFT → GRPO with TopoPRM reward → TVSD compression to compact students

The key insight: reasoning processes are inherently graph-structured, not sequential. CoT is merely a linear serialization of a DAG. By recovering the topology, we enable dense process supervision without any annotation cost.

## Artifacts

| Type | URL |
|------|-----|
| Checkpoints | https://huggingface.co/rwlinno/topoprm-ckpts |
| Data | https://huggingface.co/datasets/rwlinno/topoprm-data |
| Code | https://github.com/RWLinno/TopoPRM |

## Quick Start

```bash
# Clone and setup
git clone git@github.com:RWLinno/TopoPRM.git && cd TopoPRM
git checkout exp_May8

# Environment
conda create -n topoprm python=3.12 -y && conda activate topoprm
pip install -r requirements.txt

# Verify
python -c "from src.reward.topo_reward import TopoReward; print('OK')"
```

See [`docs/HANDOFF.md`](docs/HANDOFF.md) for complete setup instructions including data download and experiment continuation.

## Project Structure

```
topoprm/
├── src/                    # Core library
│   ├── dag/                # DAG extraction, compression, graph ops
│   ├── reward/             # Reward modules (topo, continuity, composite, ablations)
│   ├── eval/               # Unified evaluation, DAG metrics
│   ├── data/               # Data loading, DAG construction
│   └── distill/            # TVSD distillation
├── scripts/                # Runnable scripts (train, eval, data prep, analysis)
├── configs/                # YAML configs (reference)
├── topoprm_paper/          # LaTeX paper source
├── docs/                   # Documentation + HANDOFF.md
├── data/                   # .gitignored; download via setup.sh
├── output/                 # .gitignored; checkpoints + eval results
└── tests/                  # pytest
```

## Key Results

| Metric | TopoPRM (GRPO) | Outcome-Only | Delta |
|--------|---------------|-------------|-------|
| Overall Critique Acc | 29.7% | 16.3% | +13.4 |
| Format Compliance | 94.6% | 89.0% | +5.6 |
| Avg Response Length | 364 tok | 410 tok | -11.2% |

## Reward Design

| Component | Signal | Weight | Source |
|-----------|--------|--------|--------|
| Outcome | Answer correctness | 0.70 | Programmatic check |
| Format | `<think>/<answer>` compliance | 0.15 | Regex |
| Length | Token efficiency | 0.15 | Token count |
| Topology | DAG validity (acyclic, no-orphan, direction) | Multiplicative gain | Deterministic graph analysis |
| Continuity | Step traceability | Multiplicative gain | Regex-based claim matching |

Aggregation: `r_total = r_base * (1 + α * r_topo_scaled + (1-α) * r_cont_scaled)`

## Training Pipeline

```bash
# Phase 1: Build DAGs from public math datasets
python3 scripts/build_dag_public.py --datasets gsm8k math --output_dir data/dag_public

# Phase 2: Baseline evaluation
python3 scripts/bench_transformers.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
  --label baseline --benchmarks gsm8k math500 aime2024 --use_chat_template

# Phase 3a: SFT
python3 scripts/train_sft.py

# Phase 3b: GRPO with TopoPRM
python3 scripts/train_grpo.py --sft_adapter output/sft_deepseek_r1_7b/final

# Phase 4: Evaluation with PRM reranking
python3 scripts/bench_transformers.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
  --adapter output/grpo_topoprm_deepseek_r1_7b/final \
  --label topoprm --use_chat_template --sft_style
```

## GUI: DAG Visualization

```bash
bash scripts/run_dag_gui.sh
```

Streamlit interface for viewing DAG extraction, reward computation, and layer compression.

## Citation

```bibtex
@article{topoprm2026,
  title={Topology-Aware Process Rewards for Verifiable Mathematical Reasoning},
  author={Anonymous},
  journal={arXiv preprint},
  year={2026}
}
```

## License

This project is for research purposes. See LICENSE for details.
