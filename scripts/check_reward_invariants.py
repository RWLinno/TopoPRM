#!/usr/bin/env python3
"""Reward-invariant smoke test (default-behaviour regression guard).

Run this before/after any change to the reward modules to verify the
default code path produces the same rewards as the released v3b
checkpoints. All P0..P5 patches must be no-op when the corresponding
TOPO_* env vars are unset.

Stubs out MS-Swift so it runs in any env. Reloads modules cleanly.

Usage:
    python3 scripts/check_reward_invariants.py
"""
from __future__ import annotations

import os
import sys
import types
import importlib


# ---- 1. Stub MS-Swift before any project import. ----
def _install_swift_stub() -> None:
    if "swift" in sys.modules:
        return
    swift = types.ModuleType("swift")
    swift_rewards = types.ModuleType("swift.rewards")

    class ORM:  # noqa: D401
        """Minimal stub matching the MS-Swift ORM base class signature."""

        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, completions=None, solution=None, **kwargs):
            return [0.0] * (len(completions) if completions else 0)

    orms = {}
    swift_rewards.ORM = ORM
    swift_rewards.orms = orms
    swift.rewards = swift_rewards
    sys.modules["swift"] = swift
    sys.modules["swift.rewards"] = swift_rewards


_install_swift_stub()
sys.path.insert(0, "/Knowin/foundation/weilinruan/TopoPRM")


# ---- 2. Helper to reload reward_config and composite_reward together. ----
def reload_reward_modules():
    for mod in list(sys.modules):
        if mod.startswith("src.reward.") or mod.startswith("src.data."):
            del sys.modules[mod]
    if "src.reward" in sys.modules:
        del sys.modules["src.reward"]
    rc = importlib.import_module("src.reward.reward_config")
    cr = importlib.import_module("src.reward.composite_reward")
    return rc, cr


def assert_close(actual, expected, tol=1e-6, what=""):
    if abs(actual - expected) > tol:
        raise AssertionError(f"[{what}] expected {expected}, got {actual}")


def main():
    # Clear all patch flags to test defaults.
    PATCH_VARS = [
        "TOPO_RESCALE_PATCH", "TOPO_RESCALE_MIN_SPAN",
        "TOPO_HIER_AGG",
        "TOPO_CONT_REQUIRE_EVIDENCE",
        "TOPO_DAG_SENTENCE_FALLBACK", "TOPO_DAG_SENTENCE_MIN_LEN",
        "TOPO_LENGTH_UNIT", "TOPO_LENGTH_LOW", "TOPO_LENGTH_HIGH",
        "TOPO_SCAE_PRESERVE_OUTCOME", "TOPO_SCAE_FLOOR_POS", "TOPO_SCAE_FLOOR_NEG",
    ]
    for k in PATCH_VARS:
        os.environ.pop(k, None)

    rc, cr = reload_reward_modules()
    RC = rc.RewardConfig

    # ═══════════════════════════════════════════════════════════════════════
    # DEFAULT INVARIANTS (all patches OFF)
    # ═══════════════════════════════════════════════════════════════════════

    # P0 defaults
    assert RC.TOPO_RESCALE_PATCH is False
    assert RC.TOPO_RESCALE_MIN_SPAN == 0.05

    # P1 defaults
    assert RC.SCAE_PRESERVE_OUTCOME is False
    assert RC.SCAE_FLOOR_POS == 0.3
    assert RC.SCAE_FLOOR_NEG == 0.3

    # P2 defaults
    assert RC.TOPO_HIER_AGG == "additive"

    # P3 defaults
    assert RC.CONTINUITY_REQUIRE_EVIDENCE is False

    # P4 defaults
    assert RC.DAG_SENTENCE_FALLBACK is False
    assert RC.DAG_SENTENCE_MIN_LEN == 20

    # P5 defaults
    assert RC.LENGTH_UNIT == "chars"
    assert RC.LENGTH_LOW == 2000
    assert RC.LENGTH_HIGH == 4000

    print("[ok] all patch defaults preserved (P0–P5 OFF)")

    # ═══════════════════════════════════════════════════════════════════════
    # P0: _batch_rescale
    # ═══════════════════════════════════════════════════════════════════════

    THR = cr.TopoHierarchicalReward

    # Default: tiny spread → 0.5 vector
    out = THR._batch_rescale([0.5, 0.5 + 1e-9, 0.5 - 1e-9])
    assert all(abs(x - 0.5) < 1e-6 for x in out)

    # Default: normal spread → full stretch
    out = THR._batch_rescale([0.10, 0.20, 0.30])
    assert_close(out[0], 0.0, what="default lo")
    assert_close(out[1], 0.5, what="default mid")
    assert_close(out[2], 1.0, what="default hi")

    # Default: span 0.04 (> 1e-8) still stretches in default mode
    out = THR._batch_rescale([0.50, 0.52, 0.54])
    assert_close(out[0], 0.0, what="default small-span lo")
    assert_close(out[2], 1.0, what="default small-span hi")

    # Patch ON: span 0.04 < 0.05 → collapses to 0.5
    os.environ["TOPO_RESCALE_PATCH"] = "1"
    rc, cr = reload_reward_modules()
    out = cr.TopoHierarchicalReward._batch_rescale([0.50, 0.52, 0.54])
    assert all(abs(x - 0.5) < 1e-6 for x in out), f"P0 on: expected 0.5 vector, got {out}"

    # Patch ON: span 0.10 ≥ 0.05 → still stretches
    out = cr.TopoHierarchicalReward._batch_rescale([0.40, 0.50])
    assert_close(out[0], 0.0, what="P0 on large-span lo")
    assert_close(out[1], 1.0, what="P0 on large-span hi")

    os.environ.pop("TOPO_RESCALE_PATCH")
    print("[ok] P0 (_batch_rescale) default + patch verified")

    # ═══════════════════════════════════════════════════════════════════════
    # P2: TOPO_HIER_AGG toggle
    # ═══════════════════════════════════════════════════════════════════════

    os.environ["TOPO_HIER_AGG"] = "multiplicative"
    rc, _ = reload_reward_modules()
    assert rc.RewardConfig.TOPO_HIER_AGG == "multiplicative"
    os.environ.pop("TOPO_HIER_AGG")
    rc, _ = reload_reward_modules()
    assert rc.RewardConfig.TOPO_HIER_AGG == "additive"
    print("[ok] P2 (TOPO_HIER_AGG) toggle verified")

    # ═══════════════════════════════════════════════════════════════════════
    # P3: CONTINUITY_REQUIRE_EVIDENCE toggle
    # ═══════════════════════════════════════════════════════════════════════

    os.environ["TOPO_CONT_REQUIRE_EVIDENCE"] = "1"
    rc, _ = reload_reward_modules()
    assert rc.RewardConfig.CONTINUITY_REQUIRE_EVIDENCE is True
    os.environ.pop("TOPO_CONT_REQUIRE_EVIDENCE")
    rc, _ = reload_reward_modules()
    assert rc.RewardConfig.CONTINUITY_REQUIRE_EVIDENCE is False
    print("[ok] P3 (CONTINUITY_REQUIRE_EVIDENCE) toggle verified")

    # ═══════════════════════════════════════════════════════════════════════
    # P4: DAG_SENTENCE_FALLBACK toggle
    # ═══════════════════════════════════════════════════════════════════════

    os.environ["TOPO_DAG_SENTENCE_FALLBACK"] = "1"
    rc, _ = reload_reward_modules()
    assert rc.RewardConfig.DAG_SENTENCE_FALLBACK is True
    os.environ.pop("TOPO_DAG_SENTENCE_FALLBACK")
    rc, _ = reload_reward_modules()
    assert rc.RewardConfig.DAG_SENTENCE_FALLBACK is False
    print("[ok] P4 (DAG_SENTENCE_FALLBACK) toggle verified")

    # ═══════════════════════════════════════════════════════════════════════
    # P5: LENGTH_UNIT toggle + thresholds
    # ═══════════════════════════════════════════════════════════════════════

    os.environ["TOPO_LENGTH_UNIT"] = "tokens"
    os.environ["TOPO_LENGTH_LOW"] = "512"
    os.environ["TOPO_LENGTH_HIGH"] = "8192"
    rc, cr = reload_reward_modules()
    assert rc.RewardConfig.LENGTH_UNIT == "tokens"
    assert rc.RewardConfig.LENGTH_LOW == 512
    assert rc.RewardConfig.LENGTH_HIGH == 8192
    # Verify LengthReward picks up the new unit
    assert cr.LengthReward.UNIT == "tokens"
    for k in ("TOPO_LENGTH_UNIT", "TOPO_LENGTH_LOW", "TOPO_LENGTH_HIGH"):
        os.environ.pop(k)
    rc, cr = reload_reward_modules()
    assert rc.RewardConfig.LENGTH_UNIT == "chars"
    assert rc.RewardConfig.LENGTH_LOW == 2000
    assert cr.LengthReward.UNIT == "chars"
    print("[ok] P5 (LENGTH_UNIT) toggle + thresholds verified")

    # ═══════════════════════════════════════════════════════════════════════
    # P1: SCAE_PRESERVE_OUTCOME toggle
    # ═══════════════════════════════════════════════════════════════════════

    os.environ["TOPO_SCAE_PRESERVE_OUTCOME"] = "1"
    rc, _ = reload_reward_modules()
    assert rc.RewardConfig.SCAE_PRESERVE_OUTCOME is True
    os.environ.pop("TOPO_SCAE_PRESERVE_OUTCOME")
    rc, _ = reload_reward_modules()
    assert rc.RewardConfig.SCAE_PRESERVE_OUTCOME is False
    print("[ok] P1 (SCAE_PRESERVE_OUTCOME) toggle verified")

    # ═══════════════════════════════════════════════════════════════════════
    print("\n✓ All reward-invariant checks passed (P0–P5).")


if __name__ == "__main__":
    main()
