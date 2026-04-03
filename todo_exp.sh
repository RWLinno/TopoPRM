#!/usr/bin/env bash
# TopoPRM NeurIPS 2026 — Master Experiment Runbook
# 说明：每一步都用 nohup 后台运行；先看注释，再执行下一行命令。
# 更新日期：2026-04-02

# ═══════════════════════════════════════════════════════════════
# Phase 0: 环境准备
# ═══════════════════════════════════════════════════════════════

# 0a) 创建目录
mkdir -p ./logs ./output/eval ./output/analysis ./output/distill_data ./output/distill_logs

# 0b) 数据流水线 [DONE 2026-04-03]
# nohup bash scripts/run_data_pipeline.sh > ./logs/run_data_pipeline.log 2>&1 &

# 0c) 下载公开 benchmark [DONE 2026-04-03]
# nohup bash scripts/download_benchmarks.sh > ./logs/run_download_benchmarks.log 2>&1 &

# ═══════════════════════════════════════════════════════════════
# Phase 1: SFT 冷启动
# ═══════════════════════════════════════════════════════════════

# [DONE 2026-04-03] 32B SFT
# nohup bash scripts/run_sft.sh > ./logs/run_sft.log 2>&1 &

# ═══════════════════════════════════════════════════════════════
# Phase 2: GRPO 训练（主实验 + 消融）
# ═══════════════════════════════════════════════════════════════

# 2a) 主实验：层次化聚合（推荐，解决 reward collapse）
nohup bash scripts/run_grpo.sh grpo_hierarchical > ./logs/run_grpo_hierarchical.log 2>&1 &

# 2b) 消融：仅答案奖励 [DONE 2026-04-03]
# nohup bash scripts/run_grpo.sh grpo_outcome_only > ./logs/run_grpo_outcome_only.log 2>&1 &

# 2c) 消融：去掉拓扑奖励 [DONE 2026-04-03]
# nohup bash scripts/run_grpo.sh grpo_no_topo > ./logs/run_grpo_no_topo.log 2>&1 &

# 2d) 消融：去掉连续性奖励 [DONE 2026-04-03]
# nohup bash scripts/run_grpo.sh grpo_no_continuity > ./logs/run_grpo_no_continuity.log 2>&1 &

# 2e) 对比：线性聚合（用于展示 reward collapse）[DONE 2026-04-03]
# nohup bash scripts/run_grpo.sh grpo_main > ./logs/run_grpo_main.log 2>&1 &

# 2h) 新主实验：Qwen3.5-9B base + TopoPRM hierarchical（GPU 0-3）
nohup env CUDA_VISIBLE_DEVICES=0,1,2,3 NPROC_PER_NODE=4 SFT_ADAPTER=/tmp/no_adapter \
  bash scripts/run_grpo.sh grpo_hierarchical_qwen35_9b > ./logs/run_grpo_hierarchical_qwen35_9b.log 2>&1 &

# 2f) [可选] 其他聚合策略消融
# nohup bash scripts/run_grpo.sh grpo_clipped > ./logs/run_grpo_clipped.log 2>&1 &
# nohup bash scripts/run_grpo.sh grpo_confgate > ./logs/run_grpo_confgate.log 2>&1 &
# nohup bash scripts/run_grpo.sh grpo_mulgate > ./logs/run_grpo_mulgate.log 2>&1 &

# 2g) 训练监控（另开终端）
# bash scripts/monitor_training.sh 60

# ═══════════════════════════════════════════════════════════════
# Phase 3: 私有 Benchmark 评估
# ═══════════════════════════════════════════════════════════════

# 3a) 评估所有 GRPO 变体 + SFT baseline（私有批改数据集）
nohup bash scripts/run_eval_all.sh > ./logs/run_eval_all.log 2>&1 &

# ═══════════════════════════════════════════════════════════════
# Phase 4: 公开 Benchmark 评估
# ═══════════════════════════════════════════════════════════════

# 4a) TopoPRM 教师（层次化）
nohup bash scripts/run_public_benchmarks.sh Qwen/Qwen3-32B "$(find output/grpo_hierarchical -name 'checkpoint-*' -type d | sort -V | tail -1)" grpo_hierarchical > ./logs/benchmark_hierarchical.log 2>&1 &

# 4b) 外部参考模型
nohup bash scripts/run_public_benchmarks.sh Qwen/Qwen2.5-7B-Instruct "" qwen25_7b_instruct > ./logs/benchmark_qwen25_7b.log 2>&1 &
nohup bash scripts/run_public_benchmarks.sh meta-llama/Llama-3.1-8B-Instruct "" llama31_8b > ./logs/benchmark_llama31_8b.log 2>&1 &

# 4c) Base Qwen3-32B（无 adapter）
nohup bash scripts/run_public_benchmarks.sh Qwen/Qwen3-32B "" base_qwen3_32b > ./logs/benchmark_base_32b.log 2>&1 &

# ═══════════════════════════════════════════════════════════════
# Phase 5: DAG 结构质量评估 [DONE 2026-04-03]
# ═══════════════════════════════════════════════════════════════

# nohup bash scripts/run_dag_metrics.sh > ./logs/run_dag_metrics.log 2>&1 &

# ═══════════════════════════════════════════════════════════════
# Phase 6: 蒸馏
# ═══════════════════════════════════════════════════════════════

# 6a) 完整蒸馏流程（生成 -> 过滤 -> 训练 -> 评估）
nohup bash scripts/run_distill.sh distill_7b_compact > ./logs/run_distill.log 2>&1 &

# 6b) 蒸馏后公开 benchmark
# （run_distill.sh 会自动调用，也可手动执行）
# nohup bash scripts/run_public_benchmarks.sh Qwen/Qwen3-8B "$(find output/distill_7b_compact -name 'checkpoint-*' -type d | sort -V | tail -1)" distill_8b > ./logs/benchmark_distill_8b.log 2>&1 &

# ═══════════════════════════════════════════════════════════════
# Phase 7: 导出论文表格
# ═══════════════════════════════════════════════════════════════

# [DONE 2026-04-03]
# python3 -m src.eval.export_paper_tables --eval_dir output/eval --output output/eval/paper_table_summary.csv

# ═══════════════════════════════════════════════════════════════
# Phase 8: 清理（可选，先 dry-run）
# ═══════════════════════════════════════════════════════════════

# bash scripts/cleanup_experiments.sh          # dry-run
# bash scripts/cleanup_experiments.sh --execute # 实际删除
