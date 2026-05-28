# Prompt for Server A: TopoPRM/TGSD v2 Main Experiments

你是实验执行代理。当前仓库：`/Knowin/foundation/weilinruan/TopoPRM`，当前分支：`exp_May14`。  
你的唯一目标是：紧急补齐并跑通我们方法主线实验（`+TopoPRM` 与 `TGSD-Distilled`），并输出可直接用于论文主表、消融、token效率分析的数据。

---

## 0) 环境与鉴权（必须先做）

```bash
conda activate topoprm
export ALL_PROXY=http://accelerator-cname-hnpmnhnmdul3rmxrwhgend.c.vegalb.com:80
export HF_TOKEN=<YOUR_HF_TOKEN>
export WANDB_API_KEY=<YOUR_WANDB_API_KEY>
export GITHUB_TOKEN=<YOUR_GITHUB_TOKEN>
git checkout exp_May14
```

训练框架固定：`ms-swift`。  
如需网络下载模型，优先通过上述代理。

---

## 1) 方法叙事与实现约束（必须对齐）

不要随意改方法定义，必须与当前故事一致：

1. Stage 1: `SFT` 冷启动（格式与基础能力对齐）
2. Stage 2: `GRPO + TopoPRM reward`  
   - 通过 DAG 提取与打分得到拓扑相关隐式奖励 `r_topo`
   - 与格式分、长度奖惩、outcome 进行分层多源聚合
   - 使用 `SCAE` 做优势估计
3. Stage 3: `Topology-Guided On-Policy Distillation (TG-OPD)`  
   - Teacher 提供 token-level 监督
   - Student on-policy rollout 完整序列
   - 结合 RKL 完成蒸馏优化

---

## 2) 实验范围（服务器A只做“我们方法”）

基座模型（两条线都要跑）：
- `deepseek-r1-7b`
- `qwen35-9b`

变体（方法主线）：
- `+SFT`
- `+GRPO`
- `+TopoPRM`（v2 patch 全开版本）
- `TGSD-Distilled`（Stage 3）

可选补充（资源允许时）：
- `+DAPO`
- `+DPO`

---

## 3) 直接优先使用的脚本/配置

优先调用以下现成入口，不要重复造轮子：
- `scripts/run_grpo_topoprm_v2.sh`
- `configs/grpo_topoprm_v2.yaml`
- `configs/grpo_topoprm_v2.env`
- `scripts/run_tg_opd.sh`
- `scripts/check_reward_invariants.py`
- `todo_exp_ours.sh`（作为总控入口时可用）

执行顺序建议：

1. 先做不变量检查（必须）：

```bash
python3 scripts/check_reward_invariants.py
```

2. 再跑 Stage2（GRPO + TopoPRM v2）
3. 然后跑 Stage3（TG-OPD Distillation）
4. 每个阶段结束后立即触发统一评测

---

## 4) Benchmark 与指标（必须完整）

Benchmark 固定为：
- GSM8K
- MATH-500
- Olympiad
- Omni-MATH
- AIME'24
- AIME'25
- CNMO'24
- MMLU
- GPQA-D

记录指标（必须）：
- `error`
- `correct`
- `F1`
- `pass@1`（唯一硬必需）
- `pass@k`
- `maj@k`
- `prm@k`
- `#Tokens`

---

## 5) 硬约束与自动诊断

硬约束：  
我们方法（`+TopoPRM` / `TGSD-Distilled`）相对 `+SFT`、`+GRPO` 必须“优于或打平”。  
若任一关键 benchmark 出现明显劣化（默认阈值：`pass@1` 下降 > 0.5），必须自动触发：

1. 诊断报告（按 benchmark 列出差值）
2. 根因候选（reward聚合、SCAE、DAG steps、length unit、continuity gate、采样配置）
3. 可执行改进建议（给出下一轮命令，不只写文字）
4. 该配置标记为 `redo_required`

禁止掩盖劣化结果，必须诚实记录。

---

## 6) 输出物（必须落盘）

在仓库中产出：

1. `results/method_v2/leaderboard_method_v2.csv`
2. `results/method_v2/metrics_full.json`
3. `results/method_v2/train_eval_trace.md`  
   - 记录每个阶段命令、开始/结束时间、GPU分配、失败重试
4. `results/method_v2/analysis_method_vs_sft_grpo.md`  
   - 按 benchmark 标记 `win/tie/loss`
5. `results/method_v2/redo_queue.md`  
   - 需重跑项 + 推荐重跑命令

结果格式需与 baseline 窗口可直接 merge。

---

## 7) 运行管理规范

- 服务器A仅做“我们方法”训练与评测，不承担 baseline 批量评测
- 关键作业失败后自动重试一次，仍失败则记录并跳过，不阻塞全局
- 每完成一个大阶段打印简短摘要（模型、阶段、当前最佳 pass@1、待处理项）
- 最终输出一个结论段：  
  - 是否满足“优于或打平 SFT/GRPO”  
  - 不满足时，明确下一轮最小改动方案

---

## 8) 最终回复模板（执行结束时）

请按以下结构回复：

1. 完成状态总览（按模型与阶段）
2. 各 benchmark 的方法 vs SFT/GRPO 对比结论
3. 是否满足主约束（优于/打平）
4. 需要重跑的项与原因
5. 全部结果文件路径

