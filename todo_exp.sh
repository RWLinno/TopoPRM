#!/usr/bin/env bash
# TopoPRM NIPS26 experiment launch commands (one-command-per-line runbook)
# 说明：每一步都用 nohup 后台运行；先看注释，再执行下一行命令。

# 0) 准备日志目录（只需执行一次）
nohup bash -lc 'mkdir -p ./logs ./output/eval ./output/analysis ./docs/publicity/data' > ./logs/run_prepare_dirs.log 2>&1 &

# 1) 数据流水线：生成/更新训练与评测所需数据
nohup bash scripts/run_data_pipeline.sh > ./logs/run_data_pipeline.log 2>&1 &

# 2) 下载公开 benchmark 数据（MATH/GSM8K/CMATH 等）
nohup bash scripts/download_benchmarks.sh > ./logs/run_download_benchmarks.log 2>&1 &

# 3) SFT 训练：得到基础适配器（LoRA）
nohup bash scripts/run_sft.sh > ./logs/run_sft.log 2>&1 &

# 4) GRPO 主实验：TopoPRM full reward
nohup bash scripts/run_grpo.sh grpo_main > ./logs/run_grpo_main.log 2>&1 &

# 5) GRPO 消融：outcome-only
nohup bash scripts/run_grpo.sh grpo_outcome_only > ./logs/run_grpo_outcome_only.log 2>&1 &

# 6) GRPO 消融：去掉 topology reward
nohup bash scripts/run_grpo.sh grpo_no_topo > ./logs/run_grpo_no_topo.log 2>&1 &

# 7) GRPO 消融：去掉 continuity reward
nohup bash scripts/run_grpo.sh grpo_no_continuity > ./logs/run_grpo_no_continuity.log 2>&1 &

# 8) GRPO 聚合策略：clipped
nohup bash scripts/run_grpo.sh grpo_clipped > ./logs/run_grpo_clipped.log 2>&1 &

# 9) GRPO 聚合策略：confidence-gated
nohup bash scripts/run_grpo.sh grpo_confgate > ./logs/run_grpo_confgate.log 2>&1 &

# 10) GRPO 聚合策略：multiplicative-gated
nohup bash scripts/run_grpo.sh grpo_mulgate > ./logs/run_grpo_mulgate.log 2>&1 &

# 11) GRPO 聚合策略：SCAE-style（论文主方法）
nohup bash scripts/run_grpo.sh grpo_scae > ./logs/run_grpo_scae.log 2>&1 &

# 12) 优先 benchmark：不依赖 opencompass 的 light200 对比（优先拿结果）
nohup bash scripts/run_benchmark_priority.sh > ./logs/run_benchmark_priority.log 2>&1 &

# 13) 统一评测：对所有已完成模型跑 run_eval_all
nohup bash scripts/run_eval_all.sh > ./logs/run_eval_all.log 2>&1 &

# 14) 蒸馏数据生成：从 teacher 生成 distill_train.jsonl
nohup bash scripts/generate_distill_data.sh > ./logs/run_generate_distill_data.log 2>&1 &

# 15) 训练监控：持续记录 GPU/内存/共享内存
nohup bash scripts/monitor_training.sh 60 > ./logs/run_monitor_training.log 2>&1 &

# 16) 自动后处理：等待实验完成后自动评测与汇总 CSV
nohup bash scripts/auto_post_exp.sh > ./logs/run_auto_post_exp.log 2>&1 &

# 17) 生成宣传报告数据（供 docs/publicity 页面读取）
nohup python3 scripts/generate_publicity_report.py --eval_dir output/eval --output_dir docs/publicity/data > ./logs/run_generate_publicity_report.log 2>&1 &

# 18) 一体化总控（可选）：按环境变量开关控制全流程
nohup bash scripts/run_all.sh > ./logs/run_all.log 2>&1 &

# ---- 2026-03-25 临时调度记录（人工执行，不自动重启）----
# [已执行] 暂停 GPU3/5/6 上的 public eval 任务，后续暂不重启。
# [已执行] 按用户要求对 GPU0 空转执行：ps aux -> 定位 pid -> kill -9 pid。
# [结果] GPU0 残留占用 pid=2136374 在 ps aux 中不可见，kill -9 返回 No such process；
#       当前仍表现为 utilization=0 但 memory.used 较高（驱动层残留占用）。
