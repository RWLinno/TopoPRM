#!/bin/bash
# ─── TopoPRM Experiment Cleanup Script (Conservative) ───
# This script removes intermediate checkpoints and failed runs.
# Review the output before running with --execute flag.
#
# Usage:
#   bash scripts/cleanup_experiments.sh          # dry-run (default)
#   bash scripts/cleanup_experiments.sh --execute # actually delete

set -euo pipefail

DRY_RUN=true
if [[ "${1:-}" == "--execute" ]]; then
    DRY_RUN=false
    echo "⚠️  EXECUTE MODE: files will be permanently deleted!"
else
    echo "📋 DRY-RUN MODE: showing what would be deleted (use --execute to delete)"
fi
echo ""

TOTAL_FREED=0

do_rm() {
    local target="$1"
    if [ -e "$target" ]; then
        local size
        size=$(du -sh "$target" 2>/dev/null | cut -f1)
        if $DRY_RUN; then
            echo "  [would delete] $target ($size)"
        else
            rm -rf "$target"
            echo "  [deleted] $target ($size)"
        fi
    fi
}

# ─── 1. Keep only the LAST checkpoint per experiment ───
echo "=== Removing intermediate checkpoints ==="

# distill_7b_compact_rkl: keep checkpoint-500
for ckpt in output/distill_7b_compact_rkl/checkpoint-{100,200,300,400}; do
    do_rm "$ckpt"
done

# grpo_clipped: keep checkpoint-318 (last)
for ckpt in output/grpo_clipped/v2-20260324-002031/checkpoint-{150,200,250,300}; do
    do_rm "$ckpt"
done

# grpo_main: keep checkpoint-79 (last)
do_rm "output/grpo_main/v3-20260318-211524/checkpoint-50"

# grpo_no_continuity: keep checkpoint-450
for ckpt in output/grpo_no_continuity/v0-20260323-140106/checkpoint-{150,300}; do
    do_rm "$ckpt"
done

# grpo_no_topo: keep checkpoint-637
for ckpt in output/grpo_no_topo/v1-20260322-234240/checkpoint-{212,425}; do
    do_rm "$ckpt"
done

# grpo_outcome_only: keep checkpoint-212
for ckpt in output/grpo_outcome_only/v0-20260319-171419/checkpoint-{50,100}; do
    do_rm "$ckpt"
done

# sft: keep checkpoint-120 (used by GRPO) and checkpoint-50
# Note: checkpoint-120 is the adapter used by grpo_main, keep it!
do_rm "output/sft/v0-20260318-040154/checkpoint-100"

# sft_qwen3_32b: keep checkpoint-50
for ckpt in output/sft_qwen3_32b/v1-20260316-154443/checkpoint-{10,30}; do
    do_rm "$ckpt"
done

echo ""

# ─── 2. Remove failed benchmark_light runs ───
echo "=== Removing failed benchmark_light runs ==="
do_rm "output/eval/benchmark_light"

echo ""

# ─── 3. Remove empty/failed experiment dirs ───
echo "=== Removing empty/failed experiment attempts ==="
do_rm "output/grpo_confgate"
do_rm "output/grpo_scae"
do_rm "output/grpo_mulgate"

echo ""

# ─── 4. Summary ───
echo "=== Preserved files ==="
echo "  ✓ All output/eval/*_metrics.json"
echo "  ✓ All result/ directories"
echo "  ✓ output/eval/dag_metrics.json"
echo "  ✓ Last checkpoint per experiment"
echo "  ✓ output/sft/v0-20260318-040154/checkpoint-120 (GRPO adapter)"
echo ""

if $DRY_RUN; then
    echo "Run with --execute to actually delete these files."
fi
