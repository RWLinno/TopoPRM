# TopoPRM Distributed Evaluation Scripts

This directory provides a set of evaluation dispatch scripts that share a **single
contract**, so that one task manifest
([`configs/dist/eval_manifest.tsv`](../../configs/dist/eval_manifest.tsv)) can be
executed in any of the following settings:

| Setting | Entry point | Topology |
|------|------|------|
| Single node, multiple local GPUs | `scripts/dist/launch_local.sh` | `xargs -P` worker processes, one GPU per process |
| Managed Slurm cluster | `scripts/dist/submit_slurm.sbatch` | Slurm job array, one GPU per array task |
| Managed Kubernetes PyTorchJob (DDP-style custom task) | `scripts/dist/submit_k8s_pytorch.sh` | Horizontal sharding across pods; inside each pod `launch_local.sh` splits the local GPUs |
| Ray (portable across providers) | `scripts/dist/submit_ray.py` | Ray actor pool, `num_gpus=1` per task |

All four entry points ultimately call the same worker:
[`scripts/dist/run_eval_worker.sh`](run_eval_worker.sh). It reads `TASK_ID` to decide
which manifest row to execute, and pins itself to local GPU `TASK_ID % NUM_GPUS_PER_NODE`.

---

## Manifest format

Columns of [`configs/dist/eval_manifest.tsv`](../../configs/dist/eval_manifest.tsv)
(tab separated):

| Column | Meaning |
|----|------|
| `TASK_ID` | Task number (integer); every entry point uses it to locate the row |
| `LABEL` | Value passed to `bench_transformers.py --label` |
| `ADAPTER` | LoRA adapter path; `-` means no adapter |
| `SFT_STYLE` | `0`/`1`, maps to `--sft_style` |
| `BENCHMARKS` | Benchmark list joined by `+`, e.g. `aime2025+amc23+gpqa_diamond` |
| `NUM_SAMPLES` | `--num_samples_per_item` |
| `MAX_ITEMS` | `--max_items`; `0` means the full set |
| `NOTES` | Free-form text describing the purpose of the row |

Lines starting with `#` are comments and do not consume a `TASK_ID`.

When adding or modifying tasks, **edit only the manifest** — the launcher and worker
code do not need to change.

---

## 1. Running locally (recommended for a first dry-run)

```bash
# Run every task in the manifest
bash scripts/dist/launch_local.sh

# Run only 0..5 (Wave A)
bash scripts/dist/launch_local.sh 0..5

# Explicit selection (spaces or commas both work)
TASK_IDS="0 2 5" bash scripts/dist/launch_local.sh

# Cap the parallelism (default = GPU count detected via nvidia-smi)
PARALLEL=4 bash scripts/dist/launch_local.sh 0..5
```

Each task writes its stdout/stderr to `logs/dist/<LABEL>_<benches>.log`; the top-level
stdout only prints the worker summary and any failure notices.

---

## 2. Managed Slurm cluster

On a Slurm cluster that supports `sbatch --array` (including elastic resource pools),
use it directly:

```bash
sbatch --array=0-5 \
       --gres=gpu:1 \
       --cpus-per-task=8 \
       --mem=64G \
       -t 12:00:00 \
       -o logs/dist/slurm_%A_%a.log \
       scripts/dist/submit_slurm.sbatch
```

The `--array` index is exposed as `SLURM_ARRAY_TASK_ID`, which the worker reads directly
as its `TASK_ID`. Each array task requests one GPU, and Slurm handles placement across
nodes.

If the environment is managed by Conda:

```bash
sbatch --array=0-9 --gres=gpu:1 -t 24:00:00 \
       --export=ALL,CONDA_ENV=topoprm \
       scripts/dist/submit_slurm.sbatch
```

---

## 3. Managed Kubernetes PyTorchJob (DDP-style custom task)

Managed ML platforms that run PyTorch jobs on Kubernetes typically inject per-worker
environment variables such as `MLP_WORKER_NUM` / `MLP_ROLE_INDEX` / `MLP_WORKER_GPU`.
[`submit_k8s_pytorch.sh`](submit_k8s_pytorch.sh) then:

1. Splits the manifest into `MLP_WORKER_NUM` contiguous chunks;
2. Takes the chunk belonging to this worker (`MLP_ROLE_INDEX`) and fans it out over the
   local GPUs via `launch_local.sh`.

Submission template (`${CLUSTER_SUBMIT_CMD}` stands for the vendor CLI used to submit
cluster jobs, e.g. the command-line tool shipped by your managed ML platform):

```bash
${CLUSTER_SUBMIT_CMD} \
  --name topoprm_eval_dr1_7b \
  --framework PyTorchDDP \
  --task-role worker --replica 6 --gpu-per-replica 1 \
  --image ${DOCKER_REGISTRY}/<your-ns>/topoprm:latest \
  --working-dir ${REPO_ROOT} \
  --entrypoint "bash scripts/dist/submit_k8s_pytorch.sh"
```

To pin specific `TASK_ID`s to a given worker, add this to the task environment
variables:

```yaml
env:
  - name: TASK_IDS_PER_WORKER
    value: "0,3,5"
```

---

## 4. Ray (portable; recommended for elastic resource pools)

[`submit_ray.py`](submit_ray.py) is provider-agnostic — it only needs a running Ray
cluster:

```bash
# Single node (Ray + local GPUs)
ray start --head --num-gpus=8
python3 scripts/dist/submit_ray.py --task-ids 0..5

# Remote Ray cluster (e.g. Ray started on a managed elastic resource pool)
RAY_ADDRESS=ray://${RAY_HEAD_HOST}:10001 \
  python3 scripts/dist/submit_ray.py --task-ids 0..9 --num-gpus-per-task 1
```

Each Ray task requests `num_gpus=1`, and Ray schedules it onto a worker with a free GPU.
Internally it still invokes `run_eval_worker.sh` through `subprocess`, so log paths match
the local mode (`logs/dist/<LABEL>_<benches>.log`).

---

## Shared conventions

- The `MODEL_PATH` environment variable overrides the default base-model path (default:
  `${MODEL_ROOT}/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B`).
- Evaluation artifacts always land in `output/eval/<LABEL>_<BENCH>_metrics.json` and
  `*_details.jsonl`, matching the historical `bench_transformers.py` behaviour.
- `bench_transformers.py` has a built-in **skip-if-exists** rule: if `metrics.json`
  already exists, the (label, benchmark) pair is skipped by default. Pass
  `EXTRA_FLAGS="--force_overwrite"` to re-run it.
- Scripts in this directory do not write to wandb (the evaluation protocol does not
  enable wandb; see the experiment resync notes for the open TODO).
