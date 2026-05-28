# 2026-05-11 实验重整与 DR1-7B 主表补全

> 任务:接手 `di-20260414111502-4pq7b` 上的 TopoPRM 实验,清理无效进程、补齐论文主表 DR1-7B family 区块、落地分布式评测脚手架。

## 1. 接手时观察

### 1.1 机器 / 进程

- 节点:火山引擎 MLP DevInstance `di-20260414111502-4pq7b`,8×A100-80GB,CUDA 12.2,driver 535.129.03。
- 接手时正在跑 7 个 `scripts/bench_transformers.py` 评测进程(PID `3949503` `3949504` `3987623` `4098009-4098012`),其中 6 个由 5/9 启动,均已完成 GSM8K / MATH-500 / AIME'24 三个 benchmark,**全部卡在 `mmlu` benchmark 持续 0.0% acc ~12h**。
- 该死锁源自 `sft_style=True` + MCQ extractor 不兼容,详见 §3.1。
- GPU 占用:
  - 接手时 6 张卡 ~47GB / 1 张卡 ~20GB / 1 张卡空 (GPU 6)。CPU 12% / mem 3.0%。
  - 全部 7 进程在 02:25 (CST) 用 `kill` 清理后,8 卡内存全部回 0。

### 1.2 wandb 状态

- 训练阶段 wandb 记录正常:`wandb/run-20260508_164149-u0au25lw` (主 GRPO 200 步)+ 3 个 ablation run (`run-20260508_173839-*`),最后一次 `_runtime ≈ 16113s`,`train/reward ≈ 0.227`,`train/kl ≈ 0.0017`,run 终态有 `total_flos / train_runtime` 字段,落盘完整。
- **评测阶段未启用 wandb**:`scripts/bench_transformers.py` 只把 metrics 落本地 JSON(`output/eval/<label>_<bench>_metrics.json`),没有 `wandb.init` 调用。这是一个已知缺口,本次未修(详见 §6 TODO)。
- 备注:wandb run 的 `latest-run` 软链接指向 `run-20260508_173839-7sooskwo`(no_topo ablation),不是主 run;若按时间排查记得交叉确认 program 字段。

### 1.3 已完成评测数字(接手时 `output/eval/*_metrics.json`)

DR1-7B base + 5 个训后 adapter + 1 个 `topoprm_full` 标签,共 6 配置:

| 配置 / 变体                  | GSM8K | MATH-500 | AIME'24 | MMLU       |
|------------------------------|-------|----------|---------|------------|
| baseline (no adapter)        | 60.8  | 68.4     | 46.7    | **42.45**  |
| + SFT                        | 73.8  | 36.8 ⚠   | 0.0 ⚠   | —          |
| + GRPO outcome-only          | 85.1  | 67.4     | 46.7    | —          |
| + GRPO w/o topo              | 84.5  | 68.8     | 36.7    | —          |
| + GRPO w/o continuity        | 85.1  | 66.4     | 36.7    | —          |
| + TopoPRM hierarchical (200) | 84.3  | 66.6     | 50.0    | —          |
| + TopoPRM full (label dup)   | 84.5  | 66.4     | 36.7    | —          |

测评协议:`use_chat_template=True`, `max_new_tokens=8192`, `num_samples_per_item=5`, `k=[1,5]`, `batch_size=8`,answer matching 通过 `sympy.simplify` 做符号等价比较。

SFT 的 MATH-500 / AIME'24 异常低(回归),`avg_tokens=4301` 远低于其他 ~8192 — 推断 SFT trainer 在 `<answer>` tag 早停或者答案抽取走了不同分支。**留 TODO:复跑 SFT 评测**。

### 1.4 论文主表状态

- [`topoprm_paper/sections/4_experiments.tex`](../topoprm_paper/sections/4_experiments.tex) 写"DR1-7B 为主线",但 [`topoprm_paper/tables/public_results_unified.tex`](../topoprm_paper/tables/public_results_unified.tex) 当前只有 Qwen3.5-9B / Qwen2.5-7B 两个区块。
- 决策(用户选项 B):**保留 Qwen3.5-9B 区块,新增 DR1-7B family 区块**,后续 Qwen3.5-9B 仍按计划训。本次只填 DR1-7B 区块,其它列 (`AIME'25`, `CNMO'24`, `Olympiad`, `Omni-MATH`, `MMLU`, `GPQA-D`) 用 Wave A/B 跑出来回填。

## 2. 本次停损与决策

| 决策 | 动作 |
|------|------|
| MMLU eval 全 0% | **全部 kill,默认 benchmark 列表去掉 mmlu**;`scripts/bench_transformers.py` 加 `--allow_mmlu_sft_style` 兜底开关;`extract_mcq` 增加 `<answer>...</answer>` 优先匹配路径,但**不再在主表上要 MMLU 数字**(基线 42.45 保留,其它行标 `--` 含 TODO)|
| SFT AIME 0.0 | 不重跑训练,仅在 docs 记录异常并把 SFT 行单独放在主表里(让 reviewers 看到正向 vs 退化对比)|
| Wave 编排 | 先快任务 (AIME'25 + AMC23 + GPQA-D × 6 变体),再慢任务 (Olympiad + Omni-MATH × 4 关键行 × `--max_items 500`)|
| 分布式 | 不重写训练脚本,只落 eval 分布式 launcher,本机 + 阿里云 Slurm + 火山云 PyTorchDDP + Ray 四口子均通用|

## 3. 已知问题 / TODO

### 3.1 MMLU + sft_style 抽取失效

- 现象:`bench_transformers.py` 用 `--sft_style` 时,模型用 `<think>...</think><answer>...</answer>` 包推理。`extract_mcq` 早期版本未优先看 `<answer>` 内容,导致 14042 题 MMLU 全 0% acc。
- 本次修复了 `<answer>X</answer>` 路径(单字母 + boxed + "answer is X" 都已覆盖),但 MMLU 多 subject 涵盖广,需要在未来跑一遍 `--allow_mmlu_sft_style` 校验下界 acc(预期至少 30+)。
- 长期方向:把 `<answer>` 块的解析提到 `extract_number` / `extract_mcq` 之前,作为 `bench_transformers.py` 的统一前置。

### 3.2 SFT AIME / MATH-500 异常

- `output/eval/sft_dr1_7b_aime2024_metrics.json`: `pass@1=0.0`, `avg_tokens=4301`(其他配置 8192)。
- `MATH-500`: 36.8(其他 GRPO 配置 65+)。
- 假设:SFT 训练把 `<answer>` tag 学得过激,导致提前结束或答案抽取分支走错。需要复跑 SFT 评测(可能加 `--max_new_tokens 16384` + 检查 `pred_pass1` 中的 `<answer>` 实际内容)。

### 3.3 评测 wandb 落盘缺失

- `bench_transformers.py` 完全未调 wandb。要把评测数据上 wandb 需要在 `run_benchmark` 末尾加 `wandb.log(metrics)`,并在 main 启动 `wandb.init(project="topoprm-eval", name=f"{label}_{bench}")`。
- 风险:多 benchmark 在同一进程里 sequential 跑,要么每个 benchmark 一个 run,要么 `wandb.log({f"{bench}/pass@1": v})`。这次没做。

## 4. Wave A — 快任务(进行中)

### 4.1 任务清单

[`configs/dist/eval_manifest.tsv`](../configs/dist/eval_manifest.tsv) 第 0..5 行:

| TASK_ID | LABEL                  | ADAPTER                                       | sft_style | benches                         |
|---------|------------------------|-----------------------------------------------|-----------|---------------------------------|
| 0       | baseline_dr1_7b_chat   | `-`                                           | 0         | aime2025 + amc23 + gpqa_diamond |
| 1       | sft_dr1_7b             | output/sft_deepseek_r1_7b/final               | 1         | aime2025 + amc23 + gpqa_diamond |
| 2       | grpo_outcome_only      | output/grpo_outcome_only_dr1_7b/final         | 1         | aime2025 + amc23 + gpqa_diamond |
| 3       | grpo_no_topo           | output/grpo_no_topo_dr1_7b/final              | 1         | aime2025 + amc23 + gpqa_diamond |
| 4       | grpo_no_continuity     | output/grpo_no_continuity_dr1_7b/final        | 1         | aime2025 + amc23 + gpqa_diamond |
| 5       | topoprm_full_dr1_7b    | output/grpo_topoprm_deepseek_r1_7b/final      | 1         | aime2025 + amc23 + gpqa_diamond |

启动:02:55-03:05 (CST),`bash scripts/dist/launch_local.sh 0..5`。

- batch_size=8, max_new_tokens=8192, num_samples_per_item=5, k=[1,5]
- 单变体三个 benchmark sequential,共 ~6h (单卡)
- ETA:6 变体并行,wall-clock ~6h。完工时间约 09:00 (CST)。

### 4.2 进度 (待结果回填)

完成后追加此节,粘贴 `output/eval/<label>_<bench>_metrics.json` 的 pass@1 / pass@5 / maj@5 / prm@5 / avg_tokens / elapsed_sec。

| label | aime2025 p@1 | amc23 p@1 | gpqa_d p@1 | 备注 |
|-------|--------------|-----------|------------|------|
| baseline_dr1_7b_chat | _pending_ | _pending_ | _pending_ |  |
| sft_dr1_7b | _pending_ | _pending_ | _pending_ |  |
| grpo_outcome_only | _pending_ | _pending_ | _pending_ |  |
| grpo_no_topo | _pending_ | _pending_ | _pending_ |  |
| grpo_no_continuity | _pending_ | _pending_ | _pending_ |  |
| topoprm_full_dr1_7b | _pending_ | _pending_ | _pending_ |  |

## 5. Wave B — 慢任务(部分启动)

### 5.1 任务清单(`configs/dist/eval_manifest.tsv` 第 6..9 行)

| TASK_ID | LABEL                | ADAPTER                                  | benches                  | max_items | num_samples |
|---------|----------------------|------------------------------------------|--------------------------|-----------|-------------|
| 6       | baseline_dr1_7b_chat | `-`                                      | olympiadbench + omni_math | 500       | 1           |
| 7       | sft_dr1_7b           | output/sft_deepseek_r1_7b/final          | olympiadbench + omni_math | 500       | 1           |
| 8       | grpo_outcome_only    | output/grpo_outcome_only_dr1_7b/final    | olympiadbench + omni_math | 500       | 1           |
| 9       | topoprm_full_dr1_7b  | output/grpo_topoprm_deepseek_r1_7b/final | olympiadbench + omni_math | 500       | 1           |

`olympiadbench` proxy = MATH level-5 (来自 `EleutherAI/hendrycks_math` 全集 7 个 config 合并后过滤,共 1263 题,取前 500)。`omni_math` proxy = MATH level-4+5(共 2432 题,取前 500)。MATH 测试集合已落到 `data/benchmarks/MATH/test.jsonl` (4819 题,levels Counter `{5:1263, 4:1169, 3:1098, 2:860, 1:429}`)。

### 5.2 调度

- Tasks **6 / 7** 在 03:13 启动到 GPU 6 / 7(GPU 6 接手时空闲,GPU 7 是 smoke test 用过的)。
- Tasks **8 / 9** 由 [`scripts/dist/wait_and_launch.sh`](../scripts/dist/wait_and_launch.sh) 守护进程持有,轮询 GPU 0 / 1 的空闲(memory.used < 5GB)后自动启动。预期 GPU 0/1 在 Wave A baseline + SFT 完成后释放(~09:00 CST)。

### 5.3 结果(待回填)

| label | olympiadbench p@1 | omni_math p@1 | 备注 |
|-------|-------------------|---------------|------|
| baseline_dr1_7b_chat | _pending_ | _pending_ | |
| sft_dr1_7b | _pending_ | _pending_ | |
| grpo_outcome_only | _pending_ | _pending_ | |
| topoprm_full_dr1_7b | _pending_ | _pending_ | |

## 6. 主表更新约定

DR1-7B family 区块插入位置:`topoprm_paper/tables/public_results_unified.tex` 的 "Ours: 9B family (Qwen3.5-9B base)" 区块之后、"Ours: 7B family (Qwen2.5-7B base)" 之前。

Wave A 跑完后立刻把 6 行更新为真实 `aime2025` / `amc23` 数字 + GPQA-D。MMLU 列只填 baseline (42.45),其它行 `--` + 脚注「待 MCQ extractor 修复后回填」。
GSM8K / MATH-500 / AIME'24 已有的数字直接落表。

Wave B 完工后回填 Olympiad / Omni-MATH 那两列(4 关键行)。其余 SFT / 两个 ablation 行的 Olympiad / Omni-MATH 用 `--` 占位。

## 7. 分布式 evaluation 脚手架(已落地)

- 任务清单:[`configs/dist/eval_manifest.tsv`](../configs/dist/eval_manifest.tsv)
- 单一 worker:[`scripts/dist/run_eval_worker.sh`](../scripts/dist/run_eval_worker.sh)
- 4 个 launcher:
  - 本机:[`scripts/dist/launch_local.sh`](../scripts/dist/launch_local.sh)
  - 阿里云 Slurm:[`scripts/dist/submit_aliyun_slurm.sbatch`](../scripts/dist/submit_aliyun_slurm.sbatch)
  - 火山引擎 MLP PyTorchDDP:[`scripts/dist/submit_volc_pytorch.sh`](../scripts/dist/submit_volc_pytorch.sh)
  - Ray(两云通用):[`scripts/dist/submit_ray.py`](../scripts/dist/submit_ray.py)
- GPU 等待守护:[`scripts/dist/wait_and_launch.sh`](../scripts/dist/wait_and_launch.sh)
- 文档:[`scripts/dist/README.md`](../scripts/dist/README.md)

Bug 修复:`set -o pipefail` + `nvidia-smi ... | head -1` 会因 SIGPIPE 把 worker 一半概率挂掉(本次启动 Wave A 时 3/6 任务因此失败,排查后修)。所有 launcher 改用 `nvidia-smi -L | wc -l`。
