# TopoPRM 训练与复现实操（重构版）

## 目标

在不改动现有 GRPO 主流程的前提下，完成：

1. Teacher：基于 deterministic PRM 的 RLVR 训练
2. Student：基于 process-aware filtering 的 reverse-KL distillation

---

## 0) 环境

```bash
cd /mnt/users/rwl/topoprm
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
conda activate topoprm
```

安装依赖：

```bash
pip install -r requirements.txt
pip install -e .
```

---

## 1) 数据准备

```bash
bash scripts/run_data_pipeline.sh
```

输出（关键）：

- `data/sft_ready/*.jsonl`
- `data/grpo_ready/train.jsonl`
- `data/dag/*.json`

---

## 2) SFT 阶段

```bash
bash scripts/run_sft.sh
```

---

## 3) GRPO 阶段（Teacher）

推荐顺序：

```bash
bash scripts/run_grpo.sh grpo_main
bash scripts/run_grpo.sh grpo_outcome_only
bash scripts/run_grpo.sh grpo_no_topo
bash scripts/run_grpo.sh grpo_no_continuity
bash scripts/run_grpo.sh grpo_scae
```

说明：

- `grpo_main`：默认复合奖励训练
- `grpo_scae`：correctness-first shaping 的实现配置
- 若资源紧张可降 `num_generations` 与 `max_completion_length`

---

## 4) 评测阶段

```bash
bash scripts/run_eval_all.sh
python3 -m src.eval.export_paper_tables --eval_dir output/eval --output output/eval/paper_table_summary.csv
```

---

## 5) 蒸馏阶段（Student）

```bash
bash scripts/generate_distill_data.sh
# 再按 distill 配置跑 student 训练
```

建议：优先保证 teacher 轨迹质量，再进行 student 压缩。

---

## 6) 关键质量门禁

- outcome correctness 为主目标，过程奖励为结构约束
- 过程奖励可验证，不代表语义证明完备性
- 蒸馏前必须启用 process-aware filtering

---

## 7) 常见问题

- 显存/OOM：降低 `num_generations`、缩短 `max_completion_length`
- 路径问题：检查 `PYTHONPATH`
- reward 数值异常：先单测 reward 组件，再排查数据格式
