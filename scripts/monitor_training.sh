#!/bin/bash
set -euo pipefail

###############################################################################
# Monitor training progress — periodically check reward stats, loss, and
# detect reward collapse.
#
# Usage: bash scripts/monitor_training.sh [interval_seconds]
###############################################################################

INTERVAL="${1:-60}"
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "Training monitor started (interval=${INTERVAL}s). Ctrl+C to stop."
echo ""

while true; do
    echo "═══ $(date '+%Y-%m-%d %H:%M:%S') ═══"

    # Check all active GRPO experiments
    for exp_dir in output/grpo_*/; do
        [ ! -d "$exp_dir" ] && continue
        exp_name=$(basename "$exp_dir")

        # Find latest trainer_state.json
        state_file=$(find "$exp_dir" -name "trainer_state.json" -type f 2>/dev/null | sort | tail -1)
        [ -z "$state_file" ] && continue

        python3 -c "
import json, sys
with open('$state_file') as f:
    state = json.load(f)
logs = state.get('log_history', [])
if not logs:
    sys.exit(0)

last = [e for e in logs if 'reward' in e]
if not last:
    sys.exit(0)
last = last[-1]

step = last.get('step', '?')
reward = last.get('reward', 0)
reward_std = last.get('reward_std', 0)
frac_zero = last.get('frac_reward_zero_std', 0)
loss = last.get('loss', 0)
kl = last.get('kl', 0)

# Collapse detection
collapse_warn = ''
if frac_zero > 0.8:
    collapse_warn = ' ⚠️  COLLAPSE'
elif frac_zero > 0.5:
    collapse_warn = ' ⚡ HIGH'

print(f'  {\"$exp_name\":30s} step={step:>4} reward={reward:.4f} std={reward_std:.6f} zero_frac={frac_zero:.2f} loss={loss:.4f} kl={kl:.3f}{collapse_warn}')
" 2>/dev/null || true
    done

    # Check distillation
    for exp_dir in output/distill_*/; do
        [ ! -d "$exp_dir" ] && continue
        exp_name=$(basename "$exp_dir")
        state_file=$(find "$exp_dir" -name "trainer_state.json" -type f 2>/dev/null | sort | tail -1)
        [ -z "$state_file" ] && continue

        python3 -c "
import json
with open('$state_file') as f:
    state = json.load(f)
logs = state.get('log_history', [])
if not logs:
    exit(0)
last = [e for e in logs if 'loss' in e]
if not last:
    exit(0)
last = last[-1]
step = last.get('step', '?')
loss = last.get('loss', 0)
print(f'  {\"$exp_name\":30s} step={step:>4} loss={loss:.4f}')
" 2>/dev/null || true
    done

    # GPU utilization
    if command -v nvidia-smi &>/dev/null; then
        echo ""
        echo "  GPU utilization:"
        nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null | \
            while IFS=',' read -r idx util mem_used mem_total; do
                printf "    GPU%s: %3s%% util, %s/%s MiB\n" "$idx" "$util" "$mem_used" "$mem_total"
            done
    fi

    echo ""
    sleep "$INTERVAL"
done
