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

## Unified 9-Benchmark Evaluation (one-click, multi-GPU)

Run the full nine-benchmark unified protocol in a single command. The
orchestrator schedules `scripts/bench_transformers.py` across the GPUs
you hand it, runs fast benches first so the paper table gets real
numbers ASAP, retries failed benches once with reduced batch size, and
writes a live status dashboard to `logs/unified/status_<LABEL>.json`.

```bash
# Base model (foreground)
bash scripts/run_unified_eval.sh \
    /Knowin/foundation/weilinruan/hf_models/Qwen/Qwen3.5-9B "" qwen35_9b_base

# Base model (background; returns immediately)
RUN_IN_BACKGROUND=1 GPUS=1,2,3,4,5,6,7 \
    bash scripts/run_unified_eval.sh \
    /Knowin/foundation/weilinruan/hf_models/Qwen/Qwen3.5-9B "" qwen35_9b_base

# +SFT adapter (Qwen3.5-9B base + your adapter)
SFT_STYLE=1 RUN_IN_BACKGROUND=1 \
    bash scripts/run_unified_eval.sh \
    /Knowin/foundation/weilinruan/hf_models/Qwen/Qwen3.5-9B \
    output/sft_qwen35_9b/<run>/checkpoint-<N> qwen35_9b_sft

# Monitor status (refreshes every 10s, Ctrl-C to exit)
bash scripts/watch_unified_eval.sh qwen35_9b_base
```

Env overrides (all optional): `GPUS=1,2,3`, `BENCHMARKS="gsm8k math500"`,
`NUM_SAMPLES=1`, `KS=1`, `FORCE=1` (re-run even if metrics.json exists),
`SFT_STYLE=1`, `RUN_IN_BACKGROUND=1`.

**Outputs**
- Per-bench metrics: `output/eval/<LABEL>_<BENCH>_metrics.json` (`pass@1` is
  the canonical number, already a fraction).
- Per-bench details: `output/eval/<LABEL>_<BENCH>_details.jsonl`.
- Per-bench stdout/stderr: `logs/unified/<LABEL>_<BENCH>.log`.
- Orchestrator log + status: `logs/unified/orchestrator_<LABEL>_*.log` and
  `logs/unified/status_<LABEL>.json`.

**Benchmark cost ranking (Qwen3.5-9B on A100-80GB, single GPU per bench; measured 2026-05-12)**

| bench          | size  | measured wall-clock | pass@1 (base) | notes                               |
|----------------|-------|---------------------|---------------|-------------------------------------|
| aime2024       | 30    | 46 min              | 6.7%          | every sample hits 4k tokens         |
| aime2025       | 30    | 47 min              | 3.3%          | same, almost no correct answers     |
| gpqa_diamond   | 198   | 58 min              | 27.3%         | MCQ, 1536 mnt                       |
| mmlu (1500-cap)| 1500  | 2.1 h               | 52.9%         | 768 mnt; extractor updated          |
| cnmo2024 (→AMC23) | 83 | 2.2 h               | 12.0%         | AMC23 proxy, 4k mnt                 |
| gsm8k          | 1319  | ~5 h                | 67.1%         | batched, 1536 mnt                   |
| math500        | 500   | ~5 h                | 43.8%         | 3072 mnt                            |
| olympiadbench  | 500   | ≈12.9 h             | 29.8%         | 4k mnt, hardest tier                |
| omni_math      | 500   | ≈12.9 h             | 51.4%         | 4k mnt                              |

Caps are set via `BENCH_CONFIG` in
`scripts/unified_eval_orchestrator.py` to keep wall-clock bounded.

**Filling the paper table**

After one or more benches complete, fold the real numbers into
`topoprm_paper/tables/public_results_unified.tex`:

```bash
# Dry-run preview
python3 scripts/fill_paper_table.py --label qwen35_9b_base --row "Qwen3.5-9B (base)"

# Apply changes
python3 scripts/fill_paper_table.py --label qwen35_9b_base --row "Qwen3.5-9B (base)" --write
python3 scripts/fill_paper_table.py --label qwen35_9b_sft  --row "+ SFT"             --write
```

Only benches with a real `<label>_<bench>_metrics.json` are touched; the
script refuses to overwrite a cell it can't back up with a JSON.

**Cleanup**

Once results are filled, prune failed/stale logs while preserving
auditable evidence (anything with a matching `metrics.json`):

```bash
bash scripts/cleanup_unified_logs.sh            # dry-run
CONFIRM=1 bash scripts/cleanup_unified_logs.sh  # actually delete
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
