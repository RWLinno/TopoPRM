#!/usr/bin/env python3
"""Whitelist-based cleanup of training checkpoints.

Rules:
- WHITELIST: never touched; these are the checkpoints cited in the paper/evals.
- KEEP_LATEST_ONLY: within a run directory, keep only the highest-numbered
  checkpoint (this kills the intermediate `checkpoint-50`/`checkpoint-100`
  duplicates).
- FULL_RUN_DELETE: list of run directories to delete entirely (legacy 32B SFT,
  abandoned clipped/confgate/main variants, etc.).

Also supports --dry_run (default). Use --apply to actually delete.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

# Never touch these exact checkpoints. These are cited in the paper or used
# as current benchmark adapters.
WHITELIST = {
    "output/sft_qwen35_9b/v0-20260407-011328/checkpoint-626",
    "output/grpo_hierarchical_qwen35_9b_mcl4096/v2-20260407-162048/checkpoint-79",
    "output/grpo_gated_qwen35_9b_mcl4096/v4-20260407-111747/checkpoint-79",
    "output/grpo_hierarchical_qwen25_7b/v3-20260406-134423/checkpoint-318",
    "output/sft_distill_4b/v0-20260417-121952/checkpoint-2034",
    # Ablation-run final checkpoints (for paper ablation table)
    "output/grpo_no_topo_qwen35_9b_mcl4096/v1-20260407-191217/checkpoint-79",
    "output/grpo_outcome_only_qwen35_9b_mcl4096/v1-20260407-191217/checkpoint-79",
    "output/grpo_no_continuity_qwen35_9b/v0-20260404-163247/checkpoint-79",
    "output/grpo_no_topo_qwen35_9b/v0-20260404-162835/checkpoint-79",
    "output/grpo_outcome_only_qwen35_9b/v2-20260404-012842/checkpoint-79",
}

# These entire run directories are scrapped (legacy 32B/SFT duplicates, early
# failed attempts we no longer cite). Deleting saves bulk disk.
FULL_RUN_DELETE = [
    "output/sft_qwen3_32b",               # legacy 32B SFT, no longer used
    "output/sft_private_boost",           # 30-step broken SFT, 6 GB
    "output/grpo_main",                   # early abandoned baseline, 13 GB
    "output/grpo_clipped",                # early abandoned variant, 31 GB
    "output/grpo_confgate",               # early abandoned variant
    "output/grpo_mulgate",                # unused
    "output/grpo_scae",                   # unused
    "output/grpo_no_continuity",          # superseded by qwen35_9b variant
    "output/grpo_no_topo",                # superseded
    "output/grpo_outcome_only",           # superseded
    "output/distill_7b_compact_rkl",      # legacy RKL 8B student, replaced by TVSD roadmap
    "output/sft",                         # old directory, duplicates sft_qwen3_32b
    "output/grpo_hierarchical_qwen35_9b", # superseded by mcl4096 version
    "output/grpo_hierarchical_qwen35_9b_mem70",
    "output/grpo_hierarchical_qwen35_9b_ng4",
    "output/grpo_gated_qwen35_9b",        # superseded by mcl4096 v4
    "output/grpo_gated_qwen35_9b_ng4_mcl4096",
    "output/grpo_gated_qwen3_32b",        # 32B gated - never used in final paper
    "output/grpo_gated_qwen25_7b",        # superseded by qwen25_7b/v3
    "output/grpo_gated_qwen25_7b_mcl2048",
    "output/sft_qwen25_7b",               # superseded by _boost and _v2
    "output/sft_qwen25_7b_v2",            # unused
    "output/sft_qwen25_7b_boost",         # ablation, not cited
    "output/sft_qwen25_math_7b",          # explored but not used
    # Qwen2.5-7B ablation runs (only keep hierarchical)
    "output/grpo_no_continuity_qwen25_7b",
    "output/grpo_no_topo_qwen25_7b",
    "output/grpo_outcome_only_qwen25_7b",
    "output/grpo_outcome_only_qwen25_7b_mcl2048",
]


def dir_size_bytes(p: Path) -> int:
    total = 0
    for sub in p.rglob("*"):
        try:
            if sub.is_file():
                total += sub.stat().st_size
        except Exception:
            pass
    return total


def fmt_gb(b: int) -> str:
    return f"{b / (1024**3):.2f} GB"


def collect_intermediate_deletions(root: Path) -> list[Path]:
    """Within each training version dir, keep only the largest-numbered checkpoint
    unless it is in WHITELIST. Deletes older intermediates."""
    to_delete: list[Path] = []
    # Find all versioned run dirs: output/<run>/v*-*
    version_dirs = []
    for p in root.glob("*/v*-*"):
        if p.is_dir():
            version_dirs.append(p)

    for vd in version_dirs:
        # Skip if this run is scheduled for full deletion below
        if any(str(vd).startswith(str(root / Path(fr).name)) for fr in FULL_RUN_DELETE
               if Path(fr).parent == root):
            continue
        ckpts = sorted(
            (c for c in vd.iterdir() if c.is_dir() and c.name.startswith("checkpoint-")),
            key=lambda c: int(c.name.split("-")[-1]),
        )
        if len(ckpts) <= 1:
            continue
        # Keep the last one; earlier ones deletable (unless whitelisted)
        for c in ckpts[:-1]:
            if str(c) in WHITELIST:
                continue
            to_delete.append(c)
    return to_delete


def collect_full_run_deletions() -> list[Path]:
    out: list[Path] = []
    for rel in FULL_RUN_DELETE:
        p = Path(rel)
        if p.exists():
            out.append(p)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("output"))
    ap.add_argument("--apply", action="store_true",
                    help="Actually delete. Without this, dry-run only.")
    args = ap.parse_args()

    full = collect_full_run_deletions()
    inter = collect_intermediate_deletions(args.root)

    total_bytes = 0
    print("== FULL RUN DELETIONS ==")
    for p in full:
        sz = dir_size_bytes(p)
        total_bytes += sz
        print(f"  {fmt_gb(sz):>10}  {p}")
    print()
    print("== INTERMEDIATE CHECKPOINT DELETIONS ==")
    for p in inter:
        sz = dir_size_bytes(p)
        total_bytes += sz
        print(f"  {fmt_gb(sz):>10}  {p}")
    print()
    print(f"== TOTAL RECLAIMABLE: {fmt_gb(total_bytes)} ==")
    print(f"mode: {'APPLY (deleting)' if args.apply else 'DRY-RUN (nothing deleted)'}")

    if args.apply:
        for p in full + inter:
            if str(p) in WHITELIST:
                print(f"  SKIP whitelist: {p}")
                continue
            try:
                shutil.rmtree(p)
                print(f"  deleted {p}")
            except Exception as e:
                print(f"  ERROR deleting {p}: {e}")


if __name__ == "__main__":
    main()
