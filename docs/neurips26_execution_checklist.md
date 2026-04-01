# NeurIPS26 Execution Checklist (Ordered + Verifiable)

This checklist is for reproducible execution in the same order as the paper.

## 0. Environment

```bash
cd /mnt/users/rwl/topoprm
nohup conda run --no-capture-output -n topoprm python -m pip install -r requirements.txt > logs/pip_install.log 2>&1 &
```

Monitor:

```bash
tail -f logs/pip_install.log
```

## 1. Lightweight pipeline (safe default)

```bash
nohup conda run --no-capture-output -n topoprm bash scripts/run_all.sh > logs/run_all.log 2>&1 &
```

Default includes:
- benchmark collection + manifest/TBD
- DAG metrics stage (runs once dependencies are available)

Monitor:

```bash
tail -f logs/run_all.log
```

## 2. Full training/eval pipeline

```bash
RUN_SFT=1 RUN_GRPO=1 RUN_ABLATIONS=1 RUN_AGGREGATORS=1 RUN_SCAE=1 RUN_EVAL=1 \
nohup conda run --no-capture-output -n topoprm bash scripts/run_all.sh > logs/run_all.log 2>&1 &
```

## 3. Distillation stage

```bash
RUN_DISTILL=1 nohup conda run --no-capture-output -n topoprm bash scripts/run_all.sh > logs/run_all_distill.log 2>&1 &
```

## 4. Required artifacts for paper filling

- Benchmark status: `data/benchmarks/manifest.json`, `data/benchmarks/status.md`
- DAG metrics: `output/eval/dag_metrics.json`
- Critique eval metrics: `output/eval/*_metrics.json`
- Paper summary export: `output/eval/paper_table_summary.csv`
- Distill compression report: `output/eval/distill_compression.json`

## 5. Sanity checks before writing final numbers

- Confirm each stage has `Stage OK` or intentional `Stage FAIL (continue)` in logs.
- Ensure model checkpoints exist for every compared variant in `output/`.
- Ensure benchmark datasets are isolated from private train data buckets.
- Keep unresolved benchmarks in TBD list with source attempts and errors.


## 6. GPU routing + fallback (important)

When GPU 0-1 are occupied by other jobs, pin training to 2-7:

```bash
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 NPROC_PER_NODE=6 \
nohup conda run --no-capture-output -n topoprm bash scripts/run_grpo.sh grpo_main > logs/grpo_main_gpu2_7.log 2>&1 &
```

If vLLM cannot allocate KV cache on 32B, use no-vLLM fallback:

```bash
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 NPROC_PER_NODE=6 \
nohup conda run --no-capture-output -n topoprm bash scripts/train_wo_vllm.sh grpo_main > logs/grpo_main_wo_vllm_gpu2_7.log 2>&1 &
```

Monitor:

```bash
tail -f logs/grpo_main_gpu2_7.log
# or
tail -f logs/grpo_main_wo_vllm_gpu2_7.log
```

