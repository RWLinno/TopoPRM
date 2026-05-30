# Open-source models for experiments
_Exported on 2026/5/29 at GMT+8 0:42:41 from Cursor (3.2.21)_

---

**User**

需要你把常见开源模型下载到'/Knowin/foundation/weilinruan/hf_models'路径下，尤其是数学推理、空间理解、General QA、Agent常用的baseline，我需要以做实验写文章为主要目的，不求最新但主流跟经典的必须有。另外可以在hf_models中维护一个README对每个模型进行简介，比如说明出自哪里，模型在哪些方面表现好。

---

**Cursor**

我先看了下现场情况，给个快速盘点再问两个关键问题：

**当前状况**
- 目录已存在：`/Knowin/foundation/weilinruan/hf_models/`，已下载 `Qwen/Qwen3.5-4B`、`Qwen/Qwen3.5-9B`、`deepseek-ai/DeepSeek-R1-Distill-Qwen-7B`（约 42G）
- 磁盘：`/Knowin` 剩余 **44T**，完全不受限
- 网络：`huggingface.co` 直连不通，但 **`hf-mirror.com` 和 `modelscope.cn` 可达**（应使用镜像下载）
- 工具：`/Knowin/foundation/weilinruan/env/topoprm` venv 已装 `huggingface_hub 1.12.2`、`modelscope 1.36.3`、`hf` / `huggingface-cli` / `modelscope` 三个 CLI 都在
- 硬件：A100-80G × 多卡（推理/实验都不缺卡）
- 用户意图：做实验写文章，要的是"经典 + 主流"的 baseline，覆盖四个方向：数学推理、空间理解、General QA、Agent

在给正式计划之前，有两点会显著改变下载列表和规模，想请你先定一下：

> **模型规模档位？这决定总下载量（按 bf16 估）。**
> - 轻量档：主要 1B–9B，总量 ~0.3–0.5 TB，跑 baseline 最省时
> - 常规档：加上 13–14B 和 30–34B，总量 ~1.0–1.5 TB（推荐，写文章常用对照区间） ✔
> - 豪华档：再加 70B 稠密和 DeepSeek-V3/R1 等 MoE（≥3 TB，会用好几天）

> **『空间理解』这一类具体指哪种？决定是否下 VLM / 3D / 具身模型。**
> - 通用多模态 VLM（Qwen2.5-VL、InternVL、LLaVA-OneVision 等），覆盖图文空间问答
> - VLM + 空间专项（再加 SpatialVLM / SpatialRGPT / SpaceLLaVA 这类空间推理专门模型） ✔
> - 再加具身 / 3D（OpenVLA、PointLLM、SpatialBot、LLaVA-3D 等）
> - 暂时只要纯文本模型，空间理解先用空间文本 benchmark 跑，不下 VLM

信息够了，给个完整计划。

# 下载常见开源模型 baseline 到 hf_models

## 总览

目标：在 `/Knowin/foundation/weilinruan/hf_models/` 下补齐四类主流开源 baseline（数学推理、空间理解、General QA、Agent），规模档 M（1B–34B，约 1.0–1.5 TB），并维护 `README.md` 对每个模型做简介。

**现场**
- 目录已有：`Qwen/Qwen3.5-4B`、`Qwen/Qwen3.5-9B`、`deepseek-ai/DeepSeek-R1-Distill-Qwen-7B`（合计 42G），保留不动
- 磁盘：`/Knowin` 剩 44 TB，空间充裕
- 网络：`huggingface.co` 直连不通，`hf-mirror.com`（HF 镜像，200 OK）和 `modelscope.cn` 都通
- 工具：`/Knowin/foundation/weilinruan/env/topoprm` venv 已装 `huggingface_hub 1.12.2`（带 `hf` / `huggingface-cli`）和 `modelscope 1.36.3`
- 硬件：A100-80G × 多卡

**下载通道（双轨 fallback）**
- 主通道：`HF_ENDPOINT=https://hf-mirror.com` + `hf_transfer` 加速 + `hf download <repo> --local-dir hf_models/<org>/<name>`
- 副通道：镜像拿不到时（如部分 Meta/Mistral gated 仓库）改走 `modelscope download --model <org>/<name>`

## 拟下清单（按方向分组）

下表后面的 `~GB` 是 bf16 权重粗估（参考：1B≈2G）。

**数学推理（~350 GB）** — 覆盖"监督数学"和"R1 风格长思维链"两条线
- `Qwen/Qwen2.5-Math-1.5B-Instruct`（~3G）, `Qwen/Qwen2.5-Math-7B-Instruct`（~15G）— 阿里数学专项，SFT 基线
- `deepseek-ai/deepseek-math-7b-instruct`（~14G）, `deepseek-ai/deepseek-math-7b-rl`（~14G）— DeepSeek Math，GRPO 论文原作
- `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`（~3G）, `...-14B`（~28G）, `...-32B`（~64G）, `...-Llama-8B`（~16G）— R1 蒸馏家族，AIME/MATH 常用对照
- `Qwen/QwQ-32B`（~64G）— 阿里开源推理旗舰
- `Qwen/Qwen3-4B`（~8G）, `Qwen/Qwen3-8B`（~16G）, `Qwen/Qwen3-14B`（~28G）, `Qwen/Qwen3-32B`（~64G）— Qwen3 thinking，2025 主流对照

**General QA / 基础 LLM（~440 GB）**
- `meta-llama/Llama-3.1-8B-Instruct`（~16G）— 经典
- `meta-llama/Llama-3.2-1B-Instruct`（~2G）, `...-3B-Instruct`（~6G）— 小模型对照
- `mistralai/Mistral-7B-Instruct-v0.3`（~15G）— 经典
- `mistralai/Mistral-Nemo-Instruct-2407`（~24G）— 12B 主流
- `mistralai/Mixtral-8x7B-Instruct-v0.1`（~87G）— 经典 MoE
- `google/gemma-2-9b-it`（~18G）, `google/gemma-2-27b-it`（~54G）— Gemma2
- `google/gemma-3-27b-it`（~54G）— Gemma3，2025 主流
- `Qwen/Qwen2.5-7B-Instruct`（~15G）, `...-14B-Instruct`（~28G）, `...-32B-Instruct`（~64G）— Qwen2.5 三档
- `microsoft/Phi-3.5-mini-instruct`（~8G）, `microsoft/phi-4`（~28G）— 小而强

**Agent / Coding（~160 GB）**
- `Qwen/Qwen2.5-Coder-7B-Instruct`（~15G）, `...-32B-Instruct`（~64G）— 代码/Agent 常用
- `deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct`（~32G）— 16B MoE coding
- `NousResearch/Hermes-3-Llama-3.1-8B`（~16G）— Agent 微调经典
- `Salesforce/xLAM-7b-r`（~15G）— 函数调用专项
- `internlm/internlm2_5-7b-chat`（~15G）— Agent 得分高

**Spatial / VLM（~220 GB）** — 依照你选的 `vlm_spatial`
- 通用 VLM 对照：
  - `Qwen/Qwen2.5-VL-3B-Instruct`（~7G）, `...-7B-Instruct`（~16G）, `...-32B-Instruct`（~64G）
  - `OpenGVLab/InternVL3-8B`（~16G）, `...-14B`（~28G）
  - `lmms-lab/llava-onevision-qwen2-7b-ov`（~16G）
  - `lmms-lab/llama3-llava-next-8b`（~16G）
- 空间专项：
  - `remyxai/SpaceLLaVA`（13B，~26G）— SpatialVLM 开源复现
  - `a8cheng/SpatialRGPT-VILA1.5-8B`（~16G）— NVIDIA SpatialRGPT
  - `RussRobin/SpatialBot-3B`（~6G）— Phi3 底座空间推理
  - `remyxai/SpaceQwen2.5-VL-3B-Instruct`（~7G）— 小而新

**合计约

---

**User**

继续

# 工作目录与现场

Implement the plan as specified, it is attached for your reference. Do NOT edit the plan file itself.

To-do's from the plan have already been created. Do not create them again. Mark them as in_progress as you work, starting with the first one. Don't stop until you have completed all the to-dos.

---

**Cursor**

发现问题：新版 `hf` CLI 没有 `--resume-download` 参数（那是旧的 `huggingface-cli` 用法），导致主通道直接报错、全部走到慢速的 ModelScope。需要修正脚本并重跑。

下载在进行（1.6G → 2.3G），hf_transfer 的多进程下载把进度条输出抑制到日志外了。继续等。

模型文件已齐全（含 `model.safetensors`），`.incomplete` 已经清理。脚本可能卡在日志 tail 上。

烟雾测试通过：`Qwen/Qwen2.5-Math-1.5B-Instruct` 走 HF 镜像成功下载 2.9G，耗时约 25 分钟（带宽 ~1.9MB/s）。按这个速度，完整 1.17 TB 清单需要持续几天，必须后台并发。策略调整：按 4 个 category 并行（每个 category 内部串行），全部以 nohup 后台运行。

4 个 category 下载进程都已在后台启动。先趁等待时间把 README 渲染准备好（小文件下载完后再重新生成一次以更新"本地状态"）。

README 已成型。注意到 `Qwen2.5-Math-7B-Instruct` 和 `Qwen2.5-Coder-7B-Instruct` 和 `Llama-3.1-8B-Instruct` 等仓库在并行启动时 `config.json` 会立刻到位，导致 `local_status` 报"已下载"/"部分"，这是并发下载中的暂态。等所有下载完成后会重新渲染一次 README。

现在开始持续监控下载。先让脚本在后台跑约 15-20 分钟，再查状态。

