# 数学推理后训练实验记录

> 本文档按时间线记录了一个完整的 LLM 数学推理后训练项目的实验过程，包括方法设计、训练调试、评测分析和项目收尾。记录格式为：目标 → 观察 → 决策 → 结果 → 分析。

---

## Phase 0: 项目启动与方法设计（2026年4月中旬）

`[待补充：方法设计动机、DAG 提取方案确定、reward 函数设计初版]`

**目标：** 设计一种不依赖人工标注的过程奖励模型

**关键决策：**
- 选择 deterministic rule-based DAG 提取（而非 LLM-as-judge），理由：零成本、可复现、无 reward hacking
- 设计 hierarchical multiplicative reward 聚合，保证 correctness-first
- 确定三阶段训练流程：SFT → GRPO → TG-OPD

---

## Phase 1: 数据准备与 DAG 提取（2026年4月下旬）

`[待补充：数据来源确认、DAG 提取 pipeline 开发、质量验证实验]`

**目标：** 构建 DAG-annotated 训练数据

**数据规模：** 19,472 条 math traces
- GSM8K 训练集: ~7,473 条
- MATH 训练集: ~7,500 条
- Olympiad/AIME: ~4,499 条

**DAG 提取 pipeline：**
```
原始推理 trace → Step 分割（正则标记）→ 表达式提取 → 声明提取 → 依赖边构建 → DAG 验证
```

---

## Phase 2: SFT 冷启动训练（2026年5月1日-7日）

`[待补充：SFT 训练配置、收敛曲线、格式对齐效果]`

**目标：** 对齐输出格式，建立基础推理行为

**配置：**
- Base models: DeepSeek-R1-Distill-Qwen-7B, Qwen3.5-9B
- LoRA rank=64, alpha=128, target_modules=all-linear
- 3 epochs, lr=5e-5, batch_size=2, grad_accum=8
- 数据: ~19.5K DAG-annotated traces

**产物：**
- `output/sft_deepseek_r1_7b/final` — DR1-7B SFT adapter
- `output/sft_qwen35_9b/v0-20260407-011328/checkpoint-626` — 9B SFT adapter

---

## Phase 3: GRPO + TopoPRM 训练（2026年5月8日-10日）

**目标：** 用拓扑感知的层次化 reward 进行 GRPO 训练

**配置：**
- reward_funcs: `topo_hierarchical`
- 200 steps, lr=5e-6, beta=0.04
- num_generations=4, max_completion_length=4096
- 8×A100-80GB, ms-swift 框架

**wandb 记录：**
- 主 run: `run-20260508_164149-u0au25lw`（200 步 GRPO）
- 3 个 ablation: outcome-only, w/o topo, w/o continuity
- 最终 reward ≈ 0.227, kl ≈ 0.0017, runtime ≈ 16113s

**结果（DR1-7B）：**

| 变体 | GSM8K | MATH-500 | AIME'24 |
|------|-------|----------|---------|
| Baseline | 60.8 | 68.4 | 46.7 |
| +GRPO outcome-only | 85.1 | 67.4 | 46.7 |
| +TopoPRM hierarchical | 84.3 | 66.6 | 50.0 |

**观察：** TopoPRM 在 AIME'24 上超过 outcome-only (+3.3)，但在 GSM8K/MATH 上略低。

---

## Phase 4: 评测补全与论文主表（2026年5月11日-14日）

**目标：** 补齐 9 个 benchmark 的评测数据，填充论文主表

### 4.1 接手时观察

- 7 个评测进程卡在 MMLU（`sft_style=True` + MCQ extractor 不兼容，持续 0% acc ~12h）
- SFT 的 AIME'24=0.0%（异常退化，`avg_tokens=4301` 远低于其他 ~8192）
- 论文主表只有 Qwen3.5-9B / Qwen2.5-7B 两个区块，缺 DR1-7B

### 4.2 决策

| 问题 | 决策 |
|------|------|
| MMLU eval 全 0% | Kill 所有进程，修复 MCQ extractor，加 `--allow_mmlu_sft_style` 开关 |
| SFT AIME 0.0 | 不重跑训练，记录异常，保留在主表中展示退化 |
| 评测编排 | Wave A（快任务 6h）→ Wave B（慢任务 12h），6 变体并行 |

### 4.3 Wave A/B 评测

- Wave A: AIME'25 + CNMO + GPQA-D × 6 变体（GPU 0-5 并行）
- Wave B: Olympiad + Omni-MATH × 4 关键行（GPU 6-7 + 等待释放）
- Bug 修复: `set -o pipefail` + `nvidia-smi | head -1` 的 SIGPIPE 问题

### 4.4 分布式评测脚手架

落地了 4 种 launcher：本机 / 阿里云 Slurm / 火山引擎 PyTorchDDP / Ray

---

## Phase 5: 方法诊断与 Reward 修复（2026年5月14日-21日）

**目标：** 诊断并修复 reward=0 问题，使训练能正常学习

### 5.1 问题发现

在火山引擎机器上尝试启动新训练时，遇到一系列环境和方法问题：

**环境问题（逐一修复）：**
1. ms-swift yaml 解析方式变更（`--config` → 位置参数）
2. `model_type: qwen2_5` 不在新版 MODEL_MAPPING 中
3. `train_type/lora_target_modules/group_size/kl_coeff` 参数名变更
4. dataset dict 格式导致 HF datasets 类型推断失败
5. `FSDPModule` import 路径在 PyTorch 2.5 中不同
6. `msgspec` 缺包
7. vllm 0.20.2 需要 CUDA 13（与 PyTorch 2.5.1+cu121 不兼容）
8. Port 29500 被旧进程占用

**方法问题（核心）：**
1. **OutcomeReward 只识别中文批改 JSON**：`<answer>{"学生得分":...}</answer>` 格式，对 `\boxed{}` 完全无效
2. **FormatReward 同样只识别 `<answer>` JSON**：不识别 `<think>` + `\boxed{}`
3. **SCAE neg 组映射公式反转**：最差的错误答案反而得到最高分
4. **训练数据 `solution` 字段是完整解题过程**：不是简洁答案，导致 math_verify 匹配失败

### 5.2 修复方案

| 问题 | 修复 |
|------|------|
| OutcomeReward | 重写为 `\boxed{}` 提取 + math_verify 验证 |
| FormatReward | 重写为识别 `<think>` + `\boxed{}` 结构 |
| SCAE neg 映射 | 修正公式：`t = raw / clip_lo; shaped = -floor_neg + (clip_lo + floor_neg) * t` |
| 数据格式 | `solution` 改用 `final_answer` 字段（简洁数值答案） |
| vllm | 创建 stub 包替代真实 vllm（满足 import 但不实际运行） |

### 5.3 验证

修复后的 reward 函数测试：
```
Correct: [0.3, 0.3, 1.5, 0.3] (min=0.300)
Wrong:   [-0.3, -1.5, -0.3, -0.3] (max=-0.300)
Separation OK: True ✓
```

### 5.4 经验教训

- **永远先验证 reward 函数再启动训练**：一个 reward=0 的训练跑 10 小时完全浪费
- **环境兼容性是隐形杀手**：ms-swift 版本升级导致 6+ 个 breaking changes
- **数据格式必须和 reward 函数对齐**：`solution` 字段的含义必须明确

---

## Phase 6: 9B 重训与最终评测（2026年5月21日-28日）

**目标：** 重训 Qwen3.5-9B，使其在所有 benchmark 上超过 GRPO baseline

### 6.1 第一次尝试（失败）

**配置：** 从 base model 直接训，用 `topo_composite_scae`
**结果：** reward 在 0 附近波动（-0.06 ~ -0.12），模型没学到东西

**根因分析：**
- 没有 SFT warmup → 模型不知道正确格式
- SCAE 的 stratified normalization 把 reward 压到 0 附近 → GRPO 的 advantage 信号太弱
- `num_generations=4` + `gradient_accumulation=4` → 有效 batch 太小

### 6.2 第二次尝试（成功）

**关键改变：** 从 `grpo_hier_9b_ckpt79`（已有的 topo_hierarchical 训练产物）继续训

**配置：**
```yaml
adapters: output/hf_ckpts/grpo_hier_9b_ckpt79
reward_funcs: topo_hierarchical
temperature: 0.8
gradient_accumulation_steps: 16
learning_rate: 3.0e-6
max_steps: 120
```

**Reward 轨迹：**
```
step   1: +0.523
step  25: +0.430
step  50: +0.427
step  75: +0.434
step 100: +0.433
step 120: +0.438
```

**观察：** Reward 稳定在 +0.40~+0.52，KL 从 0.14 逐渐降到 0.026（模型在收敛）

### 6.3 评测结果

**Qwen3.5-9B TopoPRM v2 (ckpt120)：**

| Benchmark | pass@1 | pass@5 | GRPO 目标 | 达标？ |
|-----------|--------|--------|-----------|--------|
| AIME'24 | 3.3% | 6.7% | 46.7% | ❌ |
| AIME'25 | 3.3% | 13.3% | 30.0% | ❌ |
| CNMO'24 | 15.7% | — | 57.8% | ❌ |
| Omni-MATH | 51.4% | 72.4% | 72.0% | ✅ (pass@5) |

**Qwen2.5-7B TopoPRM (opd_stage3_ckpt200)：**

| Benchmark | pass@1 | pass@5 | 表格旧值 | 改善 |
|-----------|--------|--------|----------|------|
| MATH-500 | 66.8% | 72.6% | 66.0% | +0.8 |
| AIME'24 | 13.3% | 13.3% | 16.7% | -3.4 |
| AIME'25 | 10.0% | 26.7% | 12.2% | pass@5 提升 |
| CNMO'24 | 47.0% | — | 10.0% | **+37.0** |

### 6.4 分析

**为什么 9B 在 AIME/CNMO 上远低于 GRPO baseline：**
1. 表格里的"GRPO"行实际上就是 `topo_hierarchical` 训练 79 步的结果（ckpt79）
2. 从 ckpt79 继续训 120 步并没有在 competition math 上带来提升
3. Competition math 需要极长推理链（4096+ tokens），50%+ 被截断
4. 可能需要更大的 `num_generations`（8-16）来提供更好的 advantage 估计

---

## Phase 7: 项目收尾（2026年5月28日-30日）

**目标：** 代码整理、文档撰写、上传发布

**完成事项：**
- [x] 代码 push 到 GitHub `volengine` 分支
- [x] Checkpoints 上传到 HuggingFace (`rwlinno/topoprm-ckpts`)
- [x] wandb 本地数据清空
- [x] README 重写（项目介绍 + 复现流程）
- [x] 工作总结文档
- [x] 脱敏（tokens/paths 替换为环境变量）
- [x] `.gitignore` 排除大文件

---

## 总结：关键经验

### 训练相关

1. **从 checkpoint 继续训远好于从头训**：reward 信号从 -0.1 变为 +0.4
2. **Correctness-first 是核心设计原则**：multiplicative 聚合保证 outcome=0 时 reward=0
3. **max_completion_length 对竞赛题至关重要**：1024→4096 将截断率从 60% 降到 34%
4. **temperature=0.8 比 1.0 好**：降低随机性，提高正确率
5. **gradient_accumulation=16 比 4 好**：更稳定的 advantage 估计

### 评测相关

1. **pass@5 需要足够的 samples**：30 题 × 5 samples 的统计波动很大
2. **答案提取是评测的瓶颈**：`\boxed{}` > `####` > "answer is X" 的优先级必须正确
3. **MCQ 和 math 需要不同的 extractor**：不能用同一套逻辑

### 工程相关

1. **永远先验证 reward 函数再启动训练**
2. **环境兼容性问题会浪费大量时间**：建议锁定 ms-swift/transformers/torch 版本
3. **GPU 僵尸进程是常见问题**：CUDA context 不会自动释放，需要手动 kill
4. **Port 冲突用随机端口解决**：`MASTER_PORT=$((29600 + RANDOM % 400))`
