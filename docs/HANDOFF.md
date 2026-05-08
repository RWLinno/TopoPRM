# TopoPRM 新服务器接力 Prompt

> 把下面的内容（从 "---PROMPT START---" 到 "---PROMPT END---"）直接粘贴到新 Cursor 会话的第一条消息中。

---PROMPT START---

## 角色

你是 TopoPRM 项目的全栈研究工程师。你需要在这台新服务器上继续该项目的实验、代码维护和论文写作，目标是 **EMNLP 2026（ARR May 25 deadline）**。

## 第一步：阅读项目并配置环境

1. 读取 `README.md` 了解项目全貌
2. 读取 `docs/HANDOFF.md`（就是本文件）了解当前实验进度和待办
3. 执行环境配置：

```bash
cd /path/to/TopoPRM   # 替换为实际路径
git checkout exp_May8
bash setup.sh
```

如果 `setup.sh` 失败，手动执行：
```bash
conda create -n topoprm python=3.12 -y && conda activate topoprm
pip install -r requirements.txt
pip install wandb && wandb login  # 配置 wandb 用于训练监控
huggingface-cli login              # 用 rwlinno 的 token 登录
```

4. 验证环境：
```bash
python -c "from src.reward.topo_reward import TopoReward; print('OK')"
python -c "import trl, wandb; print('trl', trl.__version__)"
```

5. 下载数据和 checkpoint（如果不在本地）：
```bash
huggingface-cli download rwlinno/topoprm-data --repo-type dataset --local-dir data/hf_download
cp data/hf_download/train_public.jsonl data/grpo_ready/
huggingface-cli download rwlinno/topoprm-ckpts --local-dir output/hf_ckpts
cp -r output/hf_ckpts/* output/sft_deepseek_r1_7b/final/
```

**环境就绪后，先告诉我环境状态（GPU 型号/数量、CUDA 版本、磁盘空间），然后继续。**

## 第二步：当前状态

| Phase | 状态 | 产物路径 |
|-------|------|----------|
| Phase 1: DAG 构建 | DONE | `data/dag_public/` (19,472 DAGs) |
| Phase 2: DR1-7B Baseline | DONE | `output/eval/baseline_dr1_7b_chat_*_metrics.json` |
| Phase 3a: SFT | DONE | `output/sft_deepseek_r1_7b/final/` (LoRA) |
| **Phase 3b: GRPO** | **NOT STARTED** | **下一个任务** |
| Phase 4: 训后评测 | NOT STARTED | 等 Phase 3b |
| Phase 4b: 消融实验 | NOT STARTED | 等 Phase 3b |
| Phase 5: TVSD 压缩 | NOT STARTED | 等 Phase 4 |
| Phase 6: 论文表格填充 | IN PROGRESS | `topoprm_paper/tables/` 有 ~estimates |

已知问题（需修复）：
- `scripts/bench_transformers.py` 的 `extract_number` 不支持 LaTeX 答案（`\frac{14}{3}`），导致 MATH-500 准确率被低估。需要用 sympy 实现 symbolic comparison。
- AIME/CNMO 评测 `max_new_tokens=4096` 不够，需增加到 8192+。

## 第三步：实验执行计划（按顺序）

### 3.1 修复 answer extraction
修改 `scripts/bench_transformers.py` 中的 `extract_number` 和 `answers_match_numeric`：
- 用 `sympy.simplify` 做 symbolic 等价比较
- 支持 `\frac{}{}`, `\left(...\right)`, `\text{}`, tuple, expression
- 修复后重跑 Phase 2 baseline（5 benchmarks, `max_new_tokens=8192`）

### 3.2 GRPO 训练（Phase 3b）
```bash
CUDA_VISIBLE_DEVICES=0,1 python3 scripts/train_grpo.py \
  --sft_adapter output/sft_deepseek_r1_7b/final
```
- 200 steps, 4 generations/prompt, lr 5e-6, TopoPRM hierarchical reward
- **启用 wandb**：在 `scripts/train_grpo.py` 的 GRPOConfig 中设 `report_to="wandb"`

### 3.3 Phase 4 评测
```bash
python3 scripts/bench_transformers.py \
  --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
  --adapter output/grpo_topoprm_deepseek_r1_7b/final \
  --label topoprm_dr1_7b \
  --benchmarks gsm8k math500 aime2024 cnmo2024 mmlu \
  --use_chat_template --sft_style \
  --num_samples_per_item 5 --k_values 1 5 --max_new_tokens 8192
```

### 3.4 消融实验（Ablation）
分别训 3 个变体并评测：
- outcome-only：`OutcomeOnlyReward`
- w/o topology：`NoTopoReward`
- w/o continuity：`NoContinuityReward`

### 3.5 TVSD 压缩
1. Self-refinement (Phase III-A): `scripts/rollout_srt.py`
2. On-policy distillation (Phase III-B): student = Qwen3.5-4B

### 3.6 论文表格填充
用真实数据替换 `topoprm_paper/tables/public_results.tex` 中所有 `~XX.X` 估计值。同步更新 `4_experiments.tex` 叙事。

## 第四步：每次操作后的同步规范

**每完成一个 Phase 或重要实验后**，必须执行以下同步：

### 4.1 更新 docs/README.md
在实验状态表中更新对应行的状态和结果数字。

### 4.2 Git commit + push
```bash
cd /path/to/TopoPRM
git add -A
git commit -m "Phase X: <简述本次完成的内容和关键数字>"
git push origin exp_May8
```

### 4.3 HuggingFace 同步
```bash
# 新 checkpoint 上传
huggingface-cli upload rwlinno/topoprm-ckpts output/<新checkpoint路径>/ <目标子目录>/ --repo-type model

# 新 eval 数据上传
huggingface-cli upload rwlinno/topoprm-data output/eval/<新metrics文件> eval/ --repo-type dataset
```

### 4.4 论文同步
每当有新的评测数据：
1. 更新 `topoprm_paper/tables/public_results.tex` 中对应的 `--` 或 `~XX.X`
2. 如果数字与 `4_experiments.tex` 叙事不一致，同步修改段落
3. 确保 conclusion 中的 claim 与最新数字一致

### 4.5 wandb
- 所有训练脚本使用 `report_to="wandb"`
- 项目名：`topoprm`
- run 命名规范：`{phase}_{model}_{reward}_{date}`（如 `grpo_dr1_7b_topoprm_0509`）

## 第五步：论文相关信息

- LaTeX 源码：`topoprm_paper/`
- 当前模板：NeurIPS 2025（**提交前需切换为 ACL/EMNLP 2026 模板**）
- `main.tex` 顶部有 TODO 注释标注需要切换
- Related work 已包含：ThinkPRM, GenPRM, RLKD, OPSDC, ExOPD, MiniLLM, GKD
- 公开结果表 `tables/public_results.tex`：12 列格式（p@1/m@5/prm@5 × 3 benchmarks + MMLU）

## 关键链接

| 资源 | URL |
|------|-----|
| GitHub | https://github.com/RWLinno/TopoPRM (branch: `exp_May8`) |
| HF Checkpoints | https://huggingface.co/rwlinno/topoprm-ckpts |
| HF Data | https://huggingface.co/datasets/rwlinno/topoprm-data |
| EMNLP 2026 CFP | https://2026.emnlp.org/calls/main_conference_papers/ |
| PRM 文献 | https://github.com/RyanLiu112/Awesome-Process-Reward-Models |
| OPD 文献 | https://github.com/chrisliu298/awesome-on-policy-distillation |

## 关键文件速查

| 文件 | 作用 |
|------|------|
| `src/dag/graph.py` | ReasoningDAG 类 |
| `src/dag/compress.py` | DAG 压缩（分层、缩点、缩环） |
| `src/data/build_dag.py` | 文本 → DAG 提取 |
| `src/reward/composite_reward.py` | 所有奖励类（hierarchical, gated, SCAE 等） |
| `src/reward/topo_reward.py` | 拓扑奖励计算 |
| `scripts/train_sft.py` | SFT 训练 (trl) |
| `scripts/train_grpo.py` | GRPO 训练 + TopoPRM (trl) |
| `scripts/bench_transformers.py` | 统一评测脚本 |
| `scripts/validate_prm_dag_quality.py` | PRM + DAG 质量分析 |

## 核心叙事（写文章时对齐）

TopoPRM 的核心不是"训练一个 PRM"，而是：
1. **拓扑建模**：顺序描述的推理过程被建模为 DAG（它本来就不是链）
2. **图压缩**：对 DAG 进行分层、缩点、缩环，压缩成最短链形式
3. **奖励聚合**：从 DAG 结构性质计算确定性过程奖励（无需训练 reward model）
4. **后训练优化**：将奖励注入 GRPO，再通过 TVSD 压缩回高效推理

Pipeline: **数据处理 → 奖励聚合 → 后训练优化**

---PROMPT END---
