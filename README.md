# TopoPRM: Topology-Aware Process Rewards for Verifiable Mathematical Reasoning

<p align="center">
  <a href="https://github.com/RWLinno/TopoPRM"><img src="https://img.shields.io/badge/EMNLP_2026-ARR_May-blue" alt="EMNLP 2026"></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.12-green" alt="Python 3.12"></a>
  <a href="https://huggingface.co/rwlinno/topoprm-ckpts"><img src="https://img.shields.io/badge/HuggingFace-Checkpoints-yellow" alt="HF Checkpoints"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-red" alt="License"></a>
</p>

**TopoPRM** treats sequential reasoning traces as implicitly structured graphs. By recovering the topological structure of chain-of-thought (CoT) outputs, we enable dense process supervision without any annotation cost.

**TopoPRM** 将序列化的推理链建模为隐式结构图。通过恢复思维链输出的拓扑结构，实现无需人工标注的密集过程监督。

---

## Core Contributions / 核心贡献

1. **Topological Modeling / 拓扑建模**: Deterministic rule-based DAG extraction from CoT — no LLM calls, no learned models.
   确定性规则提取推理 DAG，无需 LLM 调用或训练模型。

2. **Hierarchical Reward Aggregation / 层次化奖励聚合**: Topology reward (global DAG validity) + Continuity reward (local step traceability), combined via multiplicative aggregation.
   拓扑奖励（全局 DAG 有效性）+ 连续性奖励（局部步骤可追溯性），乘法聚合。

3. **Three-Stage Pipeline / 三阶段流水线**: SFT cold-start → GRPO with TopoPRM → OPD compression to compact students.
   SFT 冷启动 → TopoPRM GRPO 训练 → OPD 蒸馏压缩。

---

## Results / 实验结果

### Qwen3.5-9B Backbone (9 Benchmarks)

| Variant | GSM8K | MATH500 | AIME24 | AIME25 | GPQA | MMLU | OlympiadBench | Omni-MATH |
|---------|-------|---------|--------|--------|------|------|---------------|-----------|
| Base | 55.2 | 19.8 | 0 | 2.2 | 12.1 | 28.6 | 11.2 | 16.4 |
| +SFT | **94.1** | **50.8** | **30.0** | **15.6** | 20.7 | **68.2** | 32.8 | 42.4 |
| +TopoPRM (Hier) | 93.5 | 49.8 | 26.7 | 12.2 | 16.7 | 61.8 | **32.8** | **42.8** |
| +TopoPRM (Gated) | 93.8 | 50.8 | 20.0 | 13.3 | **21.2** | 61.1 | 32.8 | 42.0 |
| +Outcome Only | 93.3 | 50.8 | 16.7 | 13.3 | 14.1 | 63.0 | 31.3 | 41.6 |
| +No Continuity | 51.0 | 21.6 | 6.7 | 1.1 | - | 20.2 | 8.2 | 14.9 |

### OPD Distillation Results

| Model | GSM8K | MATH500 | AIME24 | GPQA | MMLU | OlympiadBench | Omni-MATH |
|-------|-------|---------|--------|------|------|---------------|-----------|
| OPD DR1-7B | 60.6 | **60.8** | **30.0** | 34.3 | 60.3 | **46.3** | **56.9** |
| OPD Qwen2.5-7B | **91.6** | 66.0 | 10.0 | 28.3 | **69.2** | 47.8 | 57.3 |

---

## Installation / 安装

```bash
git clone https://github.com/RWLinno/TopoPRM.git && cd TopoPRM
git checkout volengine

conda create -n topoprm python=3.12 -y && conda activate topoprm
pip install -r requirements.txt

# Verify / 验证
python3 -c "from src.reward.topo_reward import TopoReward; print('OK')"
```

### Environment Variables / 环境变量

```bash
export HF_MODELS_DIR=/path/to/huggingface/models
export TOPOPRM_PYTHON=python3
```

---

## Quick Start / 快速开始

```bash
# 1. Check reward system / 检查奖励系统
python3 scripts/check_reward_invariants.py

# 2. Train with TopoPRM / TopoPRM 训练
python3 scripts/train_grpo_ablation.py --reward hierarchical \
    --output_dir output/grpo_topoprm --eval_every 20 --eval_size 32

# 3. Evaluate / 评估
python3 scripts/bench_transformers.py \
    --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
    --adapter output/grpo_topoprm/final \
    --label topoprm --benchmarks gsm8k math500 --use_chat_template
```

---

## Full Reproduction / 完整复现

See [tutorials/reproduce_experiments.md](tutorials/reproduce_experiments.md) for the complete step-by-step guide.

完整复现指南请参考 [tutorials/reproduce_experiments.md](tutorials/reproduce_experiments.md)。

### Pipeline Overview / 流水线概览

```
Phase 0: Data Preparation (DAG annotation + benchmark download)
Phase 1: SFT Cold-Start (format alignment + basic reasoning)
Phase 2: GRPO + TopoPRM (topology-aware reward training)
Phase 3: OPD Distillation (knowledge compression)
Phase 4: Unified 9-Benchmark Evaluation
```

---

## Project Structure / 项目结构

```
TopoPRM/
├── src/                        # Core library / 核心库
│   ├── dag/                    # DAG extraction & compression
│   ├── reward/                 # Reward modules (topo, continuity, composite)
│   ├── eval/                   # Evaluation & benchmark runner
│   ├── data/                   # Data loading & preparation
│   ├── distill/                # OPD/TVSD distillation
│   ├── training/               # Training utilities
│   └── gui/                    # Streamlit DAG viewer
├── scripts/                    # Runnable scripts (train, eval, data)
├── configs/                    # YAML training configs
├── tutorials/                  # Visualization & reproduction guides
├── docs/                       # Documentation & work logs
├── todo_exp.sh                 # Full experiment runbook
├── requirements.txt            # Dependencies
└── setup.py                    # Package setup
```

---

## Reward Design / 奖励设计

| Component | Signal | Source |
|-----------|--------|--------|
| Outcome | Answer correctness | Programmatic check |
| Format | `<think>/<answer>` compliance | Regex |
| Length | Token efficiency | Token count |
| Topology | DAG validity (acyclic, no-orphan) | Deterministic graph analysis |
| Continuity | Step traceability | Regex-based claim matching |

**Aggregation / 聚合公式:**

```
r_total = r_base * (1 + alpha * r_topo + (1 - alpha) * r_continuity)
```

---

## Artifacts / 资源

| Type | URL |
|------|-----|
| Code | https://github.com/RWLinno/TopoPRM |
| Checkpoints | https://huggingface.co/rwlinno/topoprm-ckpts |
| Data | https://huggingface.co/datasets/rwlinno/topoprm-data |

### Available Checkpoints / 可用模型

| Name | Base Model | Method | Steps |
|------|-----------|--------|-------|
| `sft-dr1-7b-final` | DeepSeek-R1-Distill-Qwen-7B | SFT | 3651 |
| `grpo-topoprm-dr1-7b` | DeepSeek-R1-Distill-Qwen-7B | GRPO+TopoPRM | 100 |
| `grpo-topoprm-qwen35-9b` | Qwen3.5-9B | GRPO+TopoPRM | - |
| `opd-topoprm-dr1-7b-v2` | DeepSeek-R1-Distill-Qwen-7B | OPD | 200 |
| `opd-topoprm-qwen35-9b-v2` | Qwen3.5-9B | OPD | 50 |
| `grpo-scae-qwen35-9b` | Qwen3.5-9B | GRPO+SCAE | 949 |

All adapters use LoRA (r=64, alpha=128, dropout=0.05).

---

## GUI: DAG Visualization / DAG 可视化

```bash
bash scripts/run_dag_gui.sh
```

Streamlit interface for viewing DAG extraction, reward computation, and layer compression.

---

## Citation / 引用

```bibtex
@inproceedings{topoprm2026,
  title={Topology-Aware Process Rewards for Verifiable Mathematical Reasoning},
  author={Weilin Ruan},
  booktitle={Proceedings of EMNLP 2026},
  year={2026}
}
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.
