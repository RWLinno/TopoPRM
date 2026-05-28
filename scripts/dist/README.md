# TopoPRM 分布式评测脚本

这个目录提供一套**统一契约**的评测分发脚本,允许同一份任务清单
([`configs/dist/eval_manifest.tsv`](../../configs/dist/eval_manifest.tsv))在以下场景跑起来:

| 场景 | 入口 | 拓扑 |
|------|------|------|
| 本机单节点多卡 | `scripts/dist/launch_local.sh` | `xargs -P` 多进程,每进程一卡 |
| 阿里云 PAI / 灵骏 (Slurm) | `scripts/dist/submit_aliyun_slurm.sbatch` | Slurm Job Array,每 array 任务一卡 |
| 火山引擎 MLP (PyTorch DDP custom task) | `scripts/dist/submit_volc_pytorch.sh` | 多 Pod 横向切片,每 Pod 内部再用 `launch_local.sh` 切分本地 GPU |
| Ray (两云通用) | `scripts/dist/submit_ray.py` | Ray actor 池,`num_gpus=1`/任务 |

四种入口最终都调到同一个 worker:[`scripts/dist/run_eval_worker.sh`](run_eval_worker.sh),它读 `TASK_ID` 决定执行哪一行 manifest,并把自己 pin 到 `TASK_ID % NUM_GPUS_PER_NODE` 这一张本地卡。

---

## 任务清单格式

[`configs/dist/eval_manifest.tsv`](../../configs/dist/eval_manifest.tsv) 列(Tab 分隔):

| 列 | 含义 |
|----|------|
| `TASK_ID` | 任务编号,整数,后续所有入口都靠这个找到行 |
| `LABEL` | `bench_transformers.py --label` 的值 |
| `ADAPTER` | LoRA 适配器路径;`-` 表示不加 adapter |
| `SFT_STYLE` | `0`/`1`,对应 `--sft_style` |
| `BENCHMARKS` | 用 `+` 连接的 benchmark 列表,例如 `aime2025+amc23+gpqa_diamond` |
| `NUM_SAMPLES` | `--num_samples_per_item` |
| `MAX_ITEMS` | `--max_items`,`0` 表示全集 |
| `NOTES` | 自由文本,记录用途 |

以 `#` 起头的行为注释行,不计入 TASK_ID 序列。

新增 / 修改任务时**只改 manifest**,不需要动 launcher / worker 代码。

---

## 1. 本机直接跑 (推荐先在这里 dry-run)

```bash
# 跑 manifest 中全部任务
bash scripts/dist/launch_local.sh

# 只跑 Wave A 的 0..5
bash scripts/dist/launch_local.sh 0..5

# 显式指定 (空格或逗号都行)
TASK_IDS="0 2 5" bash scripts/dist/launch_local.sh

# 限制并行度 (覆盖默认 = nvidia-smi 探测到的卡数)
PARALLEL=4 bash scripts/dist/launch_local.sh 0..5
```

每个任务的 stdout/stderr 会写到 `logs/dist/<LABEL>_<benches>.log`,顶层 stdout 只打印 worker 摘要和失败提示。

---

## 2. 阿里云 PAI / 灵骏 Slurm

阿里云 PAI 灵骏 / 弹性资源池上的 Slurm 集群(支持 `sbatch --array`)直接用:

```bash
sbatch --array=0-5 \
       --gres=gpu:1 \
       --cpus-per-task=8 \
       --mem=64G \
       -t 12:00:00 \
       -o logs/dist/slurm_%A_%a.log \
       scripts/dist/submit_aliyun_slurm.sbatch
```

`--array` 的索引会自动映射到 `SLURM_ARRAY_TASK_ID`,worker 直接读它作为 `TASK_ID`。每个 array 任务申请 1 张 GPU,Slurm 自己负责调度到不同节点。

如果环境是 Conda 管理:

```bash
sbatch --array=0-9 --gres=gpu:1 -t 24:00:00 \
       --export=ALL,CONDA_ENV=topoprm \
       scripts/dist/submit_aliyun_slurm.sbatch
```

---

## 3. 火山引擎 MLP (PyTorch DDP custom task)

火山引擎机器学习平台 (volcengine MLP) 的 PyTorch 任务会自动注入
`MLP_WORKER_NUM` / `MLP_ROLE_INDEX` / `MLP_WORKER_GPU` 等环境变量。
[`submit_volc_pytorch.sh`](submit_volc_pytorch.sh) 会:

1. 把 manifest 按 `MLP_WORKER_NUM` 平均切成若干段;
2. 取本 worker (`MLP_ROLE_INDEX`) 那段,通过 `launch_local.sh` 在本地 GPUs 上 fan-out。

提交命令模板(本地有 `volc` CLI):

```bash
volc ml_task submit \
  --name topoprm_eval_dr1_7b \
  --framework PyTorchDDP \
  --task-role worker --replica 6 --gpu-per-replica 1 \
  --image registry.cn-beijing.volces.com/<your-ns>/topoprm:latest \
  --working-dir /Knowin/foundation/weilinruan/TopoPRM \
  --entrypoint "bash scripts/dist/submit_volc_pytorch.sh"
```

要指定某个 worker 拿哪些 TASK_ID,在任务环境变量里加:

```yaml
env:
  - name: TASK_IDS_PER_WORKER
    value: "0,3,5"
```

---

## 4. Ray (两云通用,弹性资源池推荐)

[`submit_ray.py`](submit_ray.py) 不挑云,只要起好 Ray 集群即可:

```bash
# 单节点 (Ray + 本地 GPUs)
ray start --head --num-gpus=8
python3 scripts/dist/submit_ray.py --task-ids 0..5

# 远程 Ray 集群 (Aliyun / Volcengine 弹性资源池起的 Ray)
RAY_ADDRESS=ray://head.lingjun:10001 \
  python3 scripts/dist/submit_ray.py --task-ids 0..9 --num-gpus-per-task 1
```

每个 Ray 任务以 `num_gpus=1` 申请资源,Ray 自己调度到有空闲 GPU 的 worker。
内部还是 `subprocess` 调 `run_eval_worker.sh`,所以日志路径与本机模式一致
(`logs/dist/<LABEL>_<benches>.log`)。

---

## 公共约定

- `MODEL_PATH` env 变量可覆盖默认基模型路径(默认是
  `/Knowin/foundation/weilinruan/hf_models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B`)。
- 评测产物总落在 `output/eval/<LABEL>_<BENCH>_metrics.json` 和 `*_details.jsonl`,
  与历史 `bench_transformers.py` 行为一致。
- `bench_transformers.py` 内置 **skip-if-exists**:如果 `metrics.json` 已存在,
  默认跳过该 (label, benchmark) 组合。要重跑加 `EXTRA_FLAGS="--force_overwrite"`。
- 本目录脚本不写 wandb (评测协议未启用 wandb,见
  [`docs/2026-05-11-experiment-resync.md`](../../docs/2026-05-11-experiment-resync.md)
  的 TODO)。
