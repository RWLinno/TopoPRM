# TopoPRM: Topology-Aware Process Rewards for Verifiable Mathematical Reasoning

> **EMNLP 2026 (ARR May cycle)** | Qwen2.5-7B · DeepSeek-R1-7B · Qwen3.5-9B | ms-swift GRPO | Deterministic DAG Rewards

<p align="center">
  <img src="docs/assets/topoprm_overview.png" width="720" alt="TopoPRM Framework Overview"/>
</p>

## Overview

TopoPRM treats sequential reasoning traces as implicitly structured dependency graphs. Instead of relying on expensive LLM-as-judge or learned reward models, we extract **deterministic DAGs** from chain-of-thought outputs and derive dense process supervision signals from graph topology.

**Key Contributions:**
1. **Topological Reward Design** — Hierarchical multiplicative aggregation ensuring correctness-first: `r = outcome × format_gate × length_gate × topo_gain`
2. **ACE (Advantage Clipping Estimation)** — Stratified clipping that separates correct/incorrect strata, preventing reward collapse in GRPO
3. **Three-Stage Pipeline** — SFT → GRPO+TopoPRM → TG-OPD (topology-guided on-policy distillation)
4. **Zero Annotation Cost** — All rewards computed via deterministic rule-based parsing (no LLM calls)

## Artifacts

| Type | URL |
|------|-----|
| Checkpoints | [huggingface.co/rwlinno/topoprm-ckpts](https://huggingface.co/rwlinno/topoprm-ckpts) |
| Training Data | [huggingface.co/datasets/rwlinno/topoprm-data](https://huggingface.co/datasets/rwlinno/topoprm-data) |
| Code | [github.com/RWLinno/TopoPRM](https://github.com/RWLinno/TopoPRM) |

## Results

### Qwen2.5-7B + TopoPRM

| Benchmark | pass@1 | pass@5 | vs Baseline |
|-----------|--------|--------|-------------|
| MATH-500 | 66.8% | 72.6% | +0.8 |
| AIME'24 | 13.3% | 13.3% | — |
| AIME'25 | 10.0% | 26.7% | — |
| CNMO'24 | 47.0% | — | +37.0 |

### Qwen3.5-9B + TopoPRM v2

| Benchmark | pass@1 | pass@5 | Notes |
|-----------|--------|--------|-------|
| Omni-MATH | 55.2% | 76.2% | Exceeds GRPO baseline (72.0%) |
| AIME'24 | 3.3% | 6.7% | |
| AIME'25 | 3.3% | 13.3% | |
| CNMO'24 | 18.1% | — | |

### DeepSeek-R1-7B + TopoPRM

| Benchmark | pass@1 | Notes |
|-----------|--------|-------|
| GSM8K | 62.0% | |
| MATH-500 | 66.0% | |
| OlympiadBench | 58.0% | |
| Omni-MATH | 54.0% | |
| MMLU | 84.0% | |
| GPQA-D | 36.0% | |

## Quick Start

```bash
# Clone
git clone https://github.com/RWLinno/TopoPRM.git && cd TopoPRM
git checkout volengine

# Environment (Python 3.12)
conda create -n topoprm python=3.12 -y && conda activate topoprm
pip install -r requirements.txt

# Verify installation
python -c "from src.reward.composite_reward import TopoHierarchicalReward; print('TopoPRM OK')"
```

## Project Structure

```
TopoPRM/
├── src/                        # Core library
│   ├── dag/                    # DAG extraction, compression, graph operations
│   ├── reward/                 # Reward modules
│   │   ├── composite_reward.py # TopoHierarchicalReward, TopoSCAEReward (ACE)
│   │   ├── outcome_reward.py   # Math answer verification (boxed + math_verify)
│   │   ├── topo_reward.py      # DAG structure quality scoring
│   │   ├── continuity_reward.py# Reasoning step traceability
│   │   └── format_reward.py    # Format compliance (<think>/<answer>)
│   ├── eval/                   # Unified benchmark evaluation
│   ├── data/                   # Data loading, DAG construction
│   └── distill/                # TG-OPD distillation
├── configs/                    # YAML training configs (GRPO, SFT, ablations)
├── scripts/                    # Runnable scripts
│   ├── bench_transformers.py   # Unified benchmark runner (9 benchmarks)
│   ├── unified_eval_orchestrator.py  # Multi-GPU eval scheduler
│   └── build_dag_public.py     # DAG extraction from datasets
├── tests/                      # pytest test suite
├── tutorials/                  # Visualization and analysis tools
├── docs/                       # Documentation
│   ├── work_summary.md         # Complete experiment log
│   └── archived/               # Historical docs
└── results/                    # Evaluation metrics (JSON)
```

## Reward Design

| Component | Signal | Aggregation | Source |
|-----------|--------|-------------|--------|
| Outcome | Answer correctness | Base (0 or 1) | `math_verify` + `\boxed{}` extraction |
| Format | `<think>` + `\boxed{}` compliance | Gate [0.5, 1.0] | Regex |
| Length | Token efficiency | Gate [0.5, 1.0] | Token count |
| Topology | DAG validity (acyclic, directed, aligned) | Multiplicative gain | Deterministic graph analysis |
| Continuity | Step traceability | Multiplicative gain | Claim-evidence matching |

**Correctness-First Guarantee:** `outcome=0 → r_total=0` regardless of topology/format scores.

## Reproducing Experiments

### 1. Data Preparation

```bash
# Download public math datasets and extract DAGs
python scripts/build_dag_public.py \
    --datasets gsm8k math olympiadbench aime \
    --output_dir data/grpo_ready
```

### 2. Stage I — SFT

```bash
swift sft configs/sft_qwen35_9b.yaml
```

### 3. Stage II — GRPO + TopoPRM

```bash
# Set environment for TopoPRM reward
export TOPO_HIER_AGG=multiplicative
export TOPO_SCAE_PRESERVE_OUTCOME=1
export TOPO_RESCALE_PATCH=1
export TOPO_DAG_SENTENCE_FALLBACK=1
export TOPO_LENGTH_UNIT=tokens

# Train
swift rlhf configs/grpo_hierarchical_qwen35_9b_mcl4096.yaml
```

### 4. Evaluation

```bash
# Single benchmark
CUDA_VISIBLE_DEVICES=0 python scripts/bench_transformers.py \
    --model Qwen/Qwen3.5-9B \
    --adapter output/grpo_hierarchical_qwen35_9b/checkpoint-80 \
    --label topoprm_9b \
    --benchmarks aime2024 math500 gsm8k \
    --use_chat_template \
    --num_samples_per_item 5 \
    --k_values 1 5 \
    --max_new_tokens 4096

# Full 9-benchmark evaluation (multi-GPU)
bash scripts/run_unified_eval.sh \
    Qwen/Qwen3.5-9B \
    output/grpo_hierarchical_qwen35_9b/checkpoint-80 \
    topoprm_9b_final
```

### 5. Checkpoints

Download pre-trained adapters from HuggingFace:

```bash
# All checkpoints
huggingface-cli download rwlinno/topoprm-ckpts --local-dir output/hf_ckpts

# Available adapters:
#   grpo_hier_9b_ckpt79          — Qwen3.5-9B, topo_hierarchical, 79 steps
#   grpo_topoprm_dr1_7b_ckpt949 — DeepSeek-R1-7B, topo_composite_scae, 949 steps
#   opd_qwen25_7b_stage3_ckpt200 — Qwen2.5-7B-Instruct, Stage III, 200 steps
```

## Tests

```bash
pytest tests/ -v
```

## Citation

```bibtex
@inproceedings{topoprm2026,
  title={TopoPRM: Topology-Aware Process Rewards for Verifiable Mathematical Reasoning},
  author={Weilin Ruan},
  booktitle={Proceedings of EMNLP},
  year={2026}
}
```

## License

Apache 2.0. See [LICENSE](LICENSE) for details.
