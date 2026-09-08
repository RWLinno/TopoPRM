#!/bin/bash
set -euo pipefail
###############################################################################
# GRPO Training — single experiment with safety
#
# Usage: bash scripts/run_grpo.sh <config_name> [extra swift args...]
#   e.g.: bash scripts/run_grpo.sh grpo_main
#         bash scripts/run_grpo.sh grpo_outcome_only --num_train_epochs 2
#
# Env vars:
#   CUDA_VISIBLE_DEVICES  — GPUs to use (default: 0,1,2,3,4,5,6,7)
#   NPROC_PER_NODE        — number of training processes (default: 8)
#   GUARD_SHM_LIMIT_GB    — shared memory kill threshold in GB (default: 400)
#   SKIP_PREFLIGHT        — set 1 to skip GPU health check
###############################################################################

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"
source "$SCRIPT_DIR/gpu_guard.sh"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
TOPOPRM_ENV_BIN="${TOPOPRM_ENV_BIN:-/Knowin/foundation/weilinruan/env/qwen35/bin}"
if [ -x "$TOPOPRM_ENV_BIN/swift" ]; then
  export PATH="$TOPOPRM_ENV_BIN:$PATH"
fi
export DS_IGNORE_CUDA_DETECTION="${DS_IGNORE_CUDA_DETECTION:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export WANDB_PROJECT="${WANDB_PROJECT:-topoprm}"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-grpo}"
export WANDB_ENTITY="${WANDB_ENTITY:-}"
export TOPO_REWARD_LOG_EVERY="${TOPO_REWARD_LOG_EVERY:-10}"

CONFIG_NAME="${1:?Usage: $0 <config_name> [extra args...]}"
shift
CONFIG="configs/${CONFIG_NAME}.yaml"
[ ! -f "$CONFIG" ] && echo "[ERROR] Config not found: $CONFIG" && exit 1
CONFIG_ENV="${CONFIG%.yaml}.env"
if [ -f "$CONFIG_ENV" ]; then
  set -a
  source "$CONFIG_ENV"
  set +a
fi
if [ "${TOPO_DISABLE_EDGE_ENCODER:-0}" = "1" ]; then
  export TOPO_DAG_EDGE_CHECKPOINT=""
  export TOPO_DAG_EDGE_MODEL=""
  export TOPO_DAG_EDGE_REQUIRED=0
fi
if [ "${TOPO_DISABLE_CONTINUITY_SOURCE:-0}" = "1" ]; then
  export TOPO_CHOQUET_W_CONTINUITY=0
  export TOPO_CHOQUET_I_TOPOLOGY_CONTINUITY=0
fi
mkdir -p output

# Safety: env + pre-flight + cleanup
export_safe_env
[ "${SKIP_PREFLIGHT:-0}" != "1" ] && { gpu_preflight || exit 1; }
shm_cleanup

# Resume from an explicitly pinned checkpoint, or auto-resume from the latest
# checkpoint already present in the output directory.
OUTPUT_DIR="${GRPO_OUTPUT_DIR:-$(grep -E '^\s*output_dir:' "$CONFIG" | awk '{print $2}' | tr -d '"' | tr -d "'")}"
RESUME_ARG=""
if [ -n "${GRPO_RESUME_FROM_CHECKPOINT:-}" ]; then
    if [ ! -d "$GRPO_RESUME_FROM_CHECKPOINT" ] || [ ! -f "$GRPO_RESUME_FROM_CHECKPOINT/trainer_state.json" ]; then
      echo "[ERROR] Invalid explicit GRPO checkpoint: $GRPO_RESUME_FROM_CHECKPOINT" >&2
      exit 3
    fi
    RESUME_ARG="--resume_from_checkpoint $GRPO_RESUME_FROM_CHECKPOINT"
    echo "[run_grpo] Resuming from explicitly pinned checkpoint: $GRPO_RESUME_FROM_CHECKPOINT"
elif [ -n "$OUTPUT_DIR" ] && [ -d "$OUTPUT_DIR" ]; then
    LATEST=$(ls -d "${OUTPUT_DIR}"/checkpoint-* 2>/dev/null | sort -V | tail -1 || true)
    [ -n "$LATEST" ] && RESUME_ARG="--resume_from_checkpoint $LATEST" && echo "[run_grpo] Resuming from $LATEST"
fi

LOG="output/${CONFIG_NAME}_$(date +%Y%m%d_%H%M%S).log"
register_cleanup "$CONFIG_NAME"

# Auto-resolve SFT adapter to avoid stale config paths.
SFT_ADAPTER="${SFT_ADAPTER:-}"
if [ -z "$SFT_ADAPTER" ] && [ "${AUTO_RESOLVE_SFT_ADAPTER:-1}" != "0" ]; then
    SFT_ADAPTER=$(find output/sft -maxdepth 3 -name "checkpoint-*" -type d 2>/dev/null | sort -V | tail -1 || true)
fi
ADAPTER_ARG=""
if [ -n "$SFT_ADAPTER" ]; then
    if [ ! -f "$SFT_ADAPTER/adapter_config.json" ] || \
       { [ ! -f "$SFT_ADAPTER/adapter_model.safetensors" ] && [ ! -f "$SFT_ADAPTER/adapter_model.bin" ]; }; then
      echo "[ERROR] Invalid SFT adapter: $SFT_ADAPTER" >&2
      exit 3
    fi
    ADAPTER_ARG="--adapters $SFT_ADAPTER"
fi
VARIANT_ARGS=()
if [ -n "${GRPO_OUTPUT_DIR:-}" ]; then
  VARIANT_ARGS+=(--output_dir "$GRPO_OUTPUT_DIR")
fi
if [ -n "${GRPO_REWARD_FUNC:-}" ]; then
  VARIANT_ARGS+=(--reward_funcs "$GRPO_REWARD_FUNC")
fi

# Persist the effective reward environment and content hashes before launch.
# The allowlist deliberately excludes credentials and unrelated shell state.
if [ -n "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
  "$TOPOPRM_ENV_BIN/python" - \
    "$OUTPUT_DIR/run_environment.json" \
    "$PROJECT_ROOT" \
    "$CONFIG" \
    "$CONFIG_ENV" \
    "$SFT_ADAPTER" \
    "$RESUME_ARG" <<'PY'
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import yaml

output_arg, root_arg, config_arg, config_env_arg, adapter_arg, resume = sys.argv[1:7]
root = Path(root_arg).resolve()
output = Path(output_arg)
output = (root / output).resolve() if not output.is_absolute() else output.resolve()
config = Path(config_arg)
config_env = Path(config_env_arg)
adapter = Path(adapter_arg) if adapter_arg else None
if adapter is not None and not adapter.is_absolute():
    adapter = (root / adapter).resolve()
config = (root / config).resolve() if not config.is_absolute() else config.resolve()
config_env = (
    (root / config_env).resolve() if not config_env.is_absolute() else config_env.resolve()
)

safe_exact = {
    "AUTO_RESOLVE_SFT_ADAPTER",
    "CUDA_VISIBLE_DEVICES",
    "DS_IGNORE_CUDA_DETECTION",
    "GRPO_OUTPUT_DIR",
    "GRPO_RESUME_FROM_CHECKPOINT",
    "GRPO_REWARD_FUNC",
    "GUARD_GPU_LEAK_MB",
    "NPROC_PER_NODE",
    "PYTORCH_CUDA_ALLOC_CONF",
    "SFT_ADAPTER",
    "TOKENIZERS_PARALLELISM",
    "TOPOPRM_ENV_BIN",
}
safe_env = {
    key: value
    for key, value in os.environ.items()
    if key.startswith(("TOPO_", "SARL_")) or key in safe_exact
}

def file_record(path: Path) -> dict[str, int | str]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return {"bytes": path.stat().st_size, "sha256": digest.hexdigest()}

source_paths = [
    config,
    config_env,
    root / "scripts/run_grpo.sh",
    root / "scripts/run_iclr27_stage2.sh",
    root / "scripts/train_edge_encoder.py",
    root / "src/capacity_profiles.py",
    root / "src/data/build_dag.py",
    root / "src/eval/math_scoring.py",
    root / "src/reward/composite_reward.py",
    root / "src/reward/topo_reward.py",
    root / "src/reward/sarl_reward.py",
    root / "src/reward/reward_config.py",
    root / "src/reward/continuity_reward.py",
    root / "src/reward/format_reward.py",
    root / "src/reward/outcome_reward.py",
    root / "src/reward/utils.py",
    root / "src/dag/graph.py",
    root / "src/dag/node.py",
    root / "src/dag/edge_encoder.py",
]
source_paths = [path for path in source_paths if path.is_file()]

with config.open(encoding="utf-8") as handle:
    config_data = yaml.safe_load(handle) or {}
artifact_paths = []
datasets = config_data.get("dataset", [])
datasets = [datasets] if isinstance(datasets, str) else datasets
for value in datasets:
    artifact_paths.append(Path(value))
if adapter is not None:
    artifact_paths.extend(
        [adapter / "adapter_config.json", adapter / "adapter_model.safetensors"]
    )
edge_adapter = os.environ.get("TOPO_DAG_EDGE_CHECKPOINT", "")
if edge_adapter:
    edge_adapter_path = Path(edge_adapter)
    artifact_paths.extend(
        [
            edge_adapter_path / "adapter_config.json",
            edge_adapter_path / "adapter_model.safetensors",
            edge_adapter_path / "edge_encoder_config.json",
            edge_adapter_path / "metrics.json",
        ]
    )
sarl_model = os.environ.get("SARL_EMBED_MODEL", "")
if sarl_model:
    artifact_paths.extend([Path(sarl_model) / "config.json", Path(sarl_model) / "model.safetensors"])
for model_path in {
    str(config_data.get("model", "")),
    os.environ.get("TOPO_DAG_EDGE_MODEL", ""),
}:
    if model_path:
        artifact_paths.extend(
            [Path(model_path) / "config.json", Path(model_path) / "model.safetensors.index.json"]
        )
artifact_paths = [path.resolve() for path in artifact_paths if path.is_file()]

default_environment = {
    "TOPO_LAMBDA_ACYCLIC": "0.15",
    "TOPO_LAMBDA_BASE": "0.20",
    "TOPO_LAMBDA_DELTA": "0.15",
    "TOPO_LAMBDA_ORPHAN": "0.15",
    "TOPO_REQUIRE_VALID_DAG": "1",
    "TOPO_VERIFY_LOG_EVERY": "0",
}
packages = {}
for package in [
    "ms-swift",
    "torch",
    "transformers",
    "peft",
    "deepspeed",
    "safetensors",
    "scikit-learn",
    "hdbscan",
]:
    try:
        packages[package] = importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        packages[package] = None

def git_output(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=root, text=True).strip()

payload = {
    "schema_version": 1,
    "captured_at_utc": datetime.now(timezone.utc).isoformat(),
    "effective_environment": dict(sorted(safe_env.items())),
    "effective_defaults_for_unset_environment": {
        key: value for key, value in default_environment.items() if key not in safe_env
    },
    "resolved_run": {
        "adapter": str(adapter) if adapter is not None else "",
        "config": str(config),
        "output_dir": str(output.parent),
        "resume_from_checkpoint": resume,
        "reward_func": os.environ.get("GRPO_REWARD_FUNC", "from-config"),
    },
    "source_files": {str(path): file_record(path) for path in source_paths},
    "input_artifacts": {str(path): file_record(path) for path in artifact_paths},
    "packages": packages,
    "git_commit": git_output("rev-parse", "HEAD"),
    "git_dirty": bool(git_output("status", "--porcelain")),
}


def immutable_run_identity(value: dict) -> dict:
    """Drop only fields that necessarily change when the same run resumes."""
    identity = dict(value)
    identity.pop("captured_at_utc", None)
    environment = dict(identity.get("effective_environment", {}))
    environment.pop("GRPO_RESUME_FROM_CHECKPOINT", None)
    identity["effective_environment"] = environment
    resolved = dict(identity.get("resolved_run", {}))
    resolved.pop("resume_from_checkpoint", None)
    identity["resolved_run"] = resolved
    return identity


output.parent.mkdir(parents=True, exist_ok=True)
if output.is_file():
    with output.open(encoding="utf-8") as handle:
        existing = json.load(handle)
    old_identity = immutable_run_identity(existing)
    new_identity = immutable_run_identity(payload)
    if old_identity != new_identity:
        changed = sorted(
            key
            for key in set(old_identity) | set(new_identity)
            if old_identity.get(key) != new_identity.get(key)
        )
        raise RuntimeError(
            "Refusing to resume with a different canonical run identity; "
            f"changed sections: {', '.join(changed)}"
        )
    print(f"[run_grpo] Verified immutable run manifest: {output}")
else:
    fd, temporary = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    print(f"[run_grpo] Wrote immutable run manifest: {output}")
PY
fi

if [ "${GRPO_MANIFEST_ONLY:-0}" = "1" ]; then
  echo "[run_grpo] Manifest-only preflight complete; training not launched."
  exit 0
fi

echo "══════════════════════════════════════════"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] GRPO: $CONFIG_NAME"
echo "  GPUs: $CUDA_VISIBLE_DEVICES  NPROC: $NPROC_PER_NODE"
echo "  SHM limit: ${GUARD_SHM_LIMIT_GB}GB"
echo "  Adapter: ${SFT_ADAPTER:-from-config}"
echo "  Reward: ${GRPO_REWARD_FUNC:-from-config}"
echo "  Output: $OUTPUT_DIR"
if [ -n "${WANDB_API_KEY:-}" ]; then
  echo "  W&B: enabled (project=${WANDB_PROJECT}, group=${WANDB_RUN_GROUP})"
else
  echo "  W&B: WANDB_API_KEY not set, relying on existing wandb login/session"
fi
echo "  Reward component log every: ${TOPO_REWARD_LOG_EVERY} calls"
echo "  Log: $LOG"
echo "══════════════════════════════════════════"

setsid "$TOPOPRM_ENV_BIN/python" -m swift.cli.main rlhf "$CONFIG" $RESUME_ARG $ADAPTER_ARG "${VARIANT_ARGS[@]}" "$@" > >(tee "$LOG") 2>&1 &
GUARDED_PID=$!
save_pid_file "$CONFIG_NAME" "$GUARDED_PID"
start_shm_watchdog "$GUARDED_PID"
start_gpu_watchdog "$GUARDED_PID"

echo "[run_grpo] PID=$GUARDED_PID, watchdogs active."
set +e
wait "$GUARDED_PID"
EXIT_CODE=$?
set -e
GUARDED_PID=""
echo "[$(date '+%Y-%m-%d %H:%M:%S')] $CONFIG_NAME exited with code $EXIT_CODE"
exit $EXIT_CODE
