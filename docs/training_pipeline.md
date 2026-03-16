# TopoPRM Training Pipeline

End-to-end guide from raw data to a GRPO-trained critique model.

---

## Overview

```
Phase 0        Phase 1           Phase 2       Phase 3        Phase 4        Phase 5       Phase 6
Environment → Data Pipeline → SFT Training → SFT Eval → GRPO Training → GRPO Eval → Distillation
                                                                                       (placeholder)
```

---

## Phase 0: Environment Setup

### Install Dependencies

```bash
# Clone the repository
cd /mnt/users/rwl/topoprm

# Install ms-swift and project dependencies
pip install ms-swift[llm]
pip install -e .

# Verify installation
python -c "from swift.rewards import ORM; print('ms-swift ORM OK')"
python -c "from src.reward.composite_reward import TopoCompositeReward; print('TopoPRM rewards OK')"
```

### Set PYTHONPATH

Every training/evaluation command requires the project root on `PYTHONPATH` so that `src.*` imports resolve correctly. The provided shell scripts handle this automatically, but if running commands manually:

```bash
cd /mnt/users/rwl/topoprm
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
```

### Download Models

```bash
bash scripts/download_models.sh
```

This downloads the base model (Qwen3-14B) from HuggingFace/ModelScope.

### Download Benchmarks

```bash
bash scripts/download_benchmarks.sh
```

Fetches MATH, GSM8K, CMATH, GaoKao, and C-Eval-Math evaluation datasets into `data/benchmarks/`.

---

## Phase 1: Data Preparation

Run the full data pipeline with a single script:

```bash
bash scripts/run_data_pipeline.sh [RAW_INPUT_DIR]
```

Default `RAW_INPUT_DIR`: `/mnt/users/postrain/ms-swift/data/数学0305`

### Pipeline Steps

| Step | Module | Input | Output |
|---|---|---|---|
| 1. Parse raw data | `src.data.parse_raw` | Raw data directory | `data/processed/parsed.jsonl` |
| 2. Clean data | `src.data.clean` | `parsed.jsonl` | `data/processed/cleaned.jsonl` |
| 3. Build DAGs | `src.data.build_dag` | `cleaned.jsonl` | `data/dag/*.json` (one per record) |
| 4. Prepare SFT data | `src.data.prepare_sft` | `cleaned.jsonl` | `data/sft_ready/train.jsonl` |
| 5. Merge datasets | `src.data.merge_datasets` | `train.jsonl` + EN data | `data/sft_ready/train_mixed.jsonl` |
| 6. Prepare GRPO data | `src.data.prepare_grpo` | `cleaned.jsonl` + `data/dag/` | `data/grpo_ready/train.jsonl` |

### Running Steps Individually

```bash
# Step 1
python3 -m src.data.parse_raw \
    --input_dir /path/to/raw/data \
    --output_path data/processed/parsed.jsonl

# Step 2
python3 -m src.data.clean \
    --input_path data/processed/parsed.jsonl \
    --output_path data/processed/cleaned.jsonl

# Step 3
python3 -m src.data.build_dag \
    --input_path data/processed/cleaned.jsonl \
    --output_dir data/dag

# Step 4
python3 -m src.data.prepare_sft \
    --input_path data/processed/cleaned.jsonl \
    --output_path data/sft_ready/train.jsonl

# Step 5
python3 -m src.data.merge_datasets \
    --zh_path data/sft_ready/train.jsonl \
    --output_path data/sft_ready/train_mixed.jsonl

# Step 6
python3 -m src.data.prepare_grpo \
    --input_path data/processed/cleaned.jsonl \
    --dag_dir data/dag \
    --output_path data/grpo_ready/train.jsonl
```

### Output Data Formats

**SFT format** (`data/sft_ready/train.jsonl`): full conversation with system + user + assistant turns.

```json
{
  "messages": [
    {"role": "system", "content": "你是一名资深、严谨、和蔼的数学老师..."},
    {"role": "user", "content": "请你批改以下学生的数学作答。\n\n【题目】..."},
    {"role": "assistant", "content": "<think>...</think><answer>{...}</answer>"}
  ]
}
```

**GRPO format** (`data/grpo_ready/train.jsonl`): prompt-only with sidecar columns.

```json
{
  "messages": [
    {"role": "system", "content": "你是一名资深、严谨、和蔼的数学老师..."},
    {"role": "user", "content": "请你批改以下学生的数学作答。\n\n【题目】..."}
  ],
  "solution": "{\"学生得分\": 8, \"结论批改\": \"部分正确\"}",
  "reference_dag": "{\"problem_id\": \"q1\", \"nodes\": [...], \"edges\": [...]}"
}
```

---

## Phase 2: SFT Training

Supervised fine-tuning using ms-swift with LoRA.

### Quick Start

```bash
bash scripts/run_sft.sh
# Equivalent to:
# swift sft --config configs/sft_qwen3_14b.yaml
```

### Configuration Highlights (`configs/sft_qwen3_14b.yaml`)

| Parameter | Value | Notes |
|---|---|---|
| `model` | `Qwen/Qwen3-14B` | Base model |
| `train_type` | `lora` | LoRA fine-tuning |
| `lora_rank` | 64 | |
| `lora_alpha` | 128 | α/r = 2 |
| `lora_target_modules` | `all-linear` | Apply LoRA to all linear layers |
| `deepspeed` | `zero3` | ZeRO Stage 3 for memory efficiency |
| `per_device_train_batch_size` | 2 | |
| `gradient_accumulation_steps` | 8 | Effective batch = 2 × 8 × num_gpus |
| `learning_rate` | 1e-4 | |
| `num_train_epochs` | 3 | |
| `max_length` | 8192 | Max sequence length |
| `bf16` | true | BF16 mixed precision |
| `output_dir` | `output/sft_qwen3_14b` | Checkpoints saved here |

### Custom Config

```bash
bash scripts/run_sft.sh configs/my_custom_sft.yaml
```

### Expected Output

- Checkpoints in `output/sft_qwen3_14b/checkpoint-*/`
- Best model at `output/sft_qwen3_14b/best_model/`
- WandB logs (if `report_to: wandb` is set)

---

## Phase 3: SFT Evaluation

Evaluate the SFT checkpoint on math benchmarks.

### Quick Start

```bash
bash scripts/run_eval.sh
# Equivalent to:
# python3 -m src.eval.benchmark_runner --config configs/eval_config.yaml
```

### Supported Benchmarks

| Benchmark | Dataset key | Description |
|---|---|---|
| MATH | `math` | Competition-level math (English) |
| GSM8K | `gsm8k` | Grade-school math (English) |
| CMATH | `cmath` | Chinese elementary math |
| GaoKao | `gaokao` | Chinese college entrance exam |
| C-Eval-Math | `ceval_math` | C-Eval math subset |

### Selective Evaluation

```bash
python3 -m src.eval.benchmark_runner \
    --model output/sft_qwen3_14b/best_model \
    --benchmarks MATH GSM8K \
    --output_dir output/eval_results
```

### Evaluation Config (`configs/eval_config.yaml`)

The config evaluates three model checkpoints for comparison:

| Label | Path |
|---|---|
| Qwen3-14B (base) | `Qwen/Qwen3-14B` |
| Qwen3-14B + SFT | `output/sft_qwen3_14b/best_model` |
| Qwen3-14B + SFT + GRPO | `output/grpo_qwen3_14b/best_model` |

Results are written to `output/eval_results/` as JSON and markdown.

---

## Phase 4: GRPO Training

Group Relative Policy Optimisation with the TopoPRM composite reward.

### Quick Start

```bash
bash scripts/run_grpo.sh
# Equivalent to:
# swift rlhf --rlhf_type grpo --config configs/grpo_qwen3_14b.yaml
```

### Configuration Highlights (`configs/grpo_qwen3_14b.yaml`)

| Parameter | Value | Notes |
|---|---|---|
| `model` | `output/sft_qwen3_14b/best_model` | SFT checkpoint as starting point |
| `rlhf_type` | `grpo` | |
| `external_plugins` | `src/reward/composite_reward.py` | Loads custom reward code |
| `reward_funcs` | `[topo_composite]` | Registered ORM name |
| `num_generations` | 8 | Completions per prompt for group comparison |
| `temperature` | 0.7 | Sampling temperature |
| `max_completion_length` | 4096 | |
| `deepspeed` | `zero2` | ZeRO Stage 2 (lighter than SFT's Stage 3) |
| `learning_rate` | 5e-6 | Lower than SFT |
| `num_train_epochs` | 1 | |
| `output_dir` | `output/grpo_qwen3_14b` | |

### How GRPO + Custom Reward Works

1. ms-swift imports `src/reward/composite_reward.py` via `external_plugins`.
2. Module-level registration (`orms["topo_composite"] = TopoCompositeReward`) makes the reward discoverable.
3. For each training prompt, GRPO generates `num_generations` completions.
4. `TopoCompositeReward.__call__` scores each completion using the 5-component reward.
5. GRPO uses group-relative advantages to update the policy.

### Custom Config

```bash
bash scripts/run_grpo.sh configs/my_custom_grpo.yaml
```

---

## Phase 5: GRPO Evaluation

Re-run benchmarks on the GRPO checkpoint to measure improvement over SFT.

```bash
python3 -m src.eval.benchmark_runner \
    --model output/grpo_qwen3_14b/best_model \
    --output_dir output/eval_results_grpo
```

Or use the full eval config which compares base / SFT / GRPO side by side:

```bash
bash scripts/run_eval.sh configs/eval_config.yaml
```

### Expected Results Structure

```
output/eval_results/
├── MATH/
│   └── metrics.json
├── GSM8K/
│   └── metrics.json
├── CMATH/
│   └── metrics.json
├── GaoKao/
│   └── metrics.json
└── C-Eval-Math/
    └── metrics.json
```

---

## Phase 6: Distillation (Placeholder)

> **Status:** Not yet implemented.

Planned approach: distil the SFT+GRPO teacher into a smaller student model (e.g., Qwen3-7B or Qwen3-1.8B) using ms-swift's distillation utilities. This phase will be documented once the implementation is complete.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'src'` | `PYTHONPATH` not set | `export PYTHONPATH="$(pwd):${PYTHONPATH:-}"` or use the provided scripts |
| `KeyError: 'topo_composite'` | `external_plugins` path wrong or not loading | Verify path is relative to CWD; check for import errors in the plugin file |
| OOM during SFT | Model too large for available GPU memory | Reduce `per_device_train_batch_size`, enable `gradient_checkpointing: true`, or use more GPUs with DeepSpeed ZeRO |
| OOM during GRPO | `num_generations` × batch doesn't fit | Reduce `num_generations` (e.g., 4 instead of 8) or reduce `max_completion_length` |
| WandB login required | `report_to: wandb` but not authenticated | Run `wandb login` or set `WANDB_API_KEY`, or change `report_to: tensorboard` |
| Slow data loading | Large JSONL file without sharding | Pre-shard the dataset or increase `dataloader_num_workers` |
| Training loss NaN | Reward function returning NaN/Inf | Add NaN guards in reward code; check for division by zero |
| `JSONDecodeError` in reward | Malformed `solution` or `reference_dag` in dataset | Re-run data pipeline; add try/except in reward `__call__` |
| DAG files missing for GRPO | `build_dag` step was skipped or failed partially | Re-run `python3 -m src.data.build_dag`; records without DAGs use empty `{}` |
| Checkpoint not found for GRPO | SFT `best_model` path doesn't exist | Verify SFT completed successfully; check `output/sft_qwen3_14b/best_model/` |
| `swift` command not found | ms-swift not installed or not on PATH | `pip install ms-swift[llm]`; verify with `swift --help` |
| eval hangs on a sample | Generation timeout too high or model in a loop | Set `timeout_per_sample` in eval config; reduce `max_new_tokens`; set `repetition_penalty > 1.0` |

---

## Quick Reference: Full Pipeline Commands

```bash
cd /mnt/users/rwl/topoprm
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

# Phase 1: Data
bash scripts/run_data_pipeline.sh /path/to/raw/data

# Phase 2: SFT
bash scripts/run_sft.sh

# Phase 3: SFT Eval
bash scripts/run_eval.sh

# Phase 4: GRPO
bash scripts/run_grpo.sh

# Phase 5: GRPO Eval
python3 -m src.eval.benchmark_runner \
    --model output/grpo_qwen3_14b/best_model \
    --output_dir output/eval_results_grpo
```
