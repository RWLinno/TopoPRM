# Prompt for Server B: Baseline Parallel Evaluation and Supplementary Runs

你是 baseline 评测代理。当前仓库：`/Knowin/foundation/weilinruan/TopoPRM`，当前分支：`exp_May14`。  
你的唯一目标是：在不影响服务器A方法主线实验的前提下，完成 baseline 与补充评测任务，并沉淀可直接合并到论文表格的数据产物。

---

## 0) 环境与鉴权

```bash
conda activate topoprm
export ALL_PROXY=http://accelerator-cname-hnpmnhnmdul3rmxrwhgend.c.vegalb.com:80
export HF_TOKEN=<YOUR_HF_TOKEN>
export WANDB_API_KEY=<YOUR_WANDB_API_KEY>
export GITHUB_TOKEN=<YOUR_GITHUB_TOKEN>
git checkout exp_May14
```

训练框架固定：`ms-swift`。

---

## 1) 你的职责边界（严格）

你只负责 baseline 与补充评测，不负责 TopoPRM/TGSD 主方法训练。  
与服务器A解耦，避免资源冲突；需要有独立任务编排脚本。

---

## 2) 必做脚本：`todo_baseline.sh`

在仓库根目录创建并维护：`todo_baseline.sh`，用于统一管理 baseline 并行评测。

脚本要求：
- 支持任务分阶段：`prepare` / `eval` / `aggregate` / `sync`
- 支持 GPU 白名单（示例：`GPU_POOL=0,1,2`），避免与服务器A抢卡
- 支持失败自动重试 1 次
- 支持日志分目录落盘（模型/变体/benchmark 粒度）
- 支持断点续跑（已完成任务自动跳过）

---

## 3) baseline 覆盖范围

基座模型：
- `deepseek-r1-7b`
- `qwen35-9b`

baseline 变体：
- `+SFT`
- `+GRPO`
- 可选：`+DAPO`
- 可选：`+DPO`

不要在该窗口跑：
- `+TopoPRM`
- `TGSD-Distilled`

---

## 4) Benchmark 与指标

Benchmark 固定：
- GSM8K
- MATH-500
- Olympiad
- Omni-MATH
- AIME'24
- AIME'25
- CNMO'24
- MMLU
- GPQA-D

指标（必须）：
- `error`
- `correct`
- `F1`
- `pass@1`
- `pass@k`
- `maj@k`
- `prm@k`
- `#Tokens`

---

## 5) 任务目标与优先级

优先级从高到低：

1. 补齐论文主表所需 baseline 缺失项（任何 `--` 或缺评测项）
2. 对已有 checkpoint 做统一协议复评（确保横向可比）
3. 资源允许时补跑 `+DAPO` / `+DPO`
4. 对异常项（如 token 异常、0分、明显偏差）做二次复核

注意：不新增不必要重训；优先“已有 checkpoint 的高质量补评测”。

---

## 6) 输出物（必须落盘）

统一写入 `results/baseline/`：

1. `results/baseline/leaderboard_baseline.csv`
2. `results/baseline/metrics_full_baseline.json`
3. `results/baseline/eval_trace_baseline.md`  
   - 包含任务调度、重试、失败原因、命令记录
4. `results/baseline/missing_cells_report.md`  
   - 标记仍缺失的 cell（模型 × 变体 × benchmark）
5. `results/baseline/merge_manifest.md`  
   - 给出与方法窗口合并所需文件路径与字段映射

格式必须与服务器A的 `results/method_v2/*` 兼容，可直接 join。

---

## 7) 并行调度建议

- 默认并发不要超过可用 GPU 数
- 长任务与短任务混排，避免尾部拖慢
- 每完成一批 benchmark 就执行一次增量聚合，减少最终失败风险
- 对单任务超时设置保护（例如 walltime 上限），超时后自动重试或降级

---

## 8) 最终回复模板（执行结束时）

请按结构输出：

1. baseline 覆盖完成率（模型 × 变体 × benchmark）
2. 缺失项清单（若有）
3. 异常项与复核结论
4. 可直接与方法结果合并的文件路径
5. 下一步建议（如需额外补跑）

