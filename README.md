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

### GUI: DAG 可视化启动（重点）

TopoPRM 提供了一个 Streamlit 可视化界面，用于查看：
- 线性推理文本如何被解析成层次化 DAG；
- 各 reward 分量（outcome/format/topology/continuity/length）如何计算；
- DAG 结构标注、层次压缩映射（layer -> compressed chain）、导出 JSON 与图片。

启动方式（推荐）：

```bash
cd /mnt/users/rwl/topoprm
conda activate topoprm

# 默认端口 8765，默认关闭 watcher（避免 inotify 上限问题）
bash scripts/run_dag_gui.sh

# 指定端口和数据集
bash scripts/run_dag_gui.sh 8765 data/grpo_ready/train.jsonl none

# 指定最多向后扫描多少个端口（默认 30）
bash scripts/run_dag_gui.sh 8765 data/grpo_ready/train.jsonl none 100
```

成功后可访问：
- `http://localhost:8765`
- `http://<你的机器IP>:8765`

单视图展示说明（默认）：
- 一个图中同时显示顺序边（solid）和依赖边（virtual/barrier）；
- 每一层节点由虚线框包裹（layer boxes）；
- 箭头终点对齐节点边缘（非圆心），并对不同边型做曲率分离，避免重叠；
- 右侧文本显示 `Lk -> Ck`，表示第 `k` 层压缩后映射到第 `k` 个链节点。

如果页面打不开，优先排查：
1. `logs/dag_gui.log` 是否有报错；
2. 端口是否被占用（换端口重启）；
3. 是否激活了正确环境（`conda activate topoprm`）。

端口占用提示：
- 当前脚本已支持自动端口回退：当 `8765` 被占用，会自动尝试 `8766`、`8767`...（可通过第四个参数控制扫描范围）。
- 启动后终端会打印最终 URL（例如 `http://localhost:8766`），按打印地址访问即可。

常见问题：
- `ModuleNotFoundError: No module named 'src'`  
  已在脚本中通过 `PYTHONPATH` 注入解决。若你手工运行 `streamlit`，请确保 `PYTHONPATH` 包含项目根目录。
- `inotify watch limit reached`  
  默认脚本已设置 `STREAMLIT_SERVER_FILE_WATCHER_TYPE=none` 规避。若需要热更新 watcher，请自行调高系统 `fs.inotify.max_user_watches`。

提取/构图相关可选环境变量：
- `TOPO_SEQ_WEAK_EDGE_MODE=adaptive|full|off`  
  - `adaptive`（默认）：只在缺少强依赖时补顺序弱边，减少“全链化”退化；
  - `full`：强制保留完整顺序链；
  - `off`：关闭顺序弱边，仅看依赖/屏障关系。
- `TOPO_ENABLE_SEQUENTIAL_WEAK_EDGE=0|1`：总开关。

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

### Formula-Level Topology Diagnostics

`r_topo` 按如下分项计算并保留逐样本诊断（可在 GUI 里查看 `topology_terms`）：

```text
r_topo = λ_b*I[|V|>0] + λ_a*I[acyclic] + λ_o*I[rho_orphan=0] + λ_d*delta + λ_k*kappa
```

诊断字段包含：
- 原始指标：`rho_orphan`, `delta`, `kappa`
- 逐项贡献：`term_base`, `term_acyclic`, `term_orphan`, `term_delta`, `term_kappa`
- 系数与归一化：`lambda_*`, `denom`, `r_topo`

当没有参考图时，`kappa` 项自动置零并从归一化分母移除（不惩罚无参考数据）。

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
