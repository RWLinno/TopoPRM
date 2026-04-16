#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Reward invariant checker -- verifies that TopoPRM core reward logic has not drifted.

Run after any code change to reward modules to ensure innovation points are intact.

Usage:
    python3 scripts/check_reward_invariants.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.reward.topo_reward import TopoReward
from src.reward.continuity_reward import ContinuityReward
from src.reward.format_reward import FormatReward
from src.reward.outcome_reward import OutcomeReward
from src.reward.composite_reward import (
    TopoHierarchicalReward,
    TopoCompositeReward,
)


def _wrap(text):
    """Wrap text as a single-element completions list."""
    return [[{"role": "assistant", "content": text}]]


def _wrap2(t1, t2):
    """Wrap two texts as a two-element completions list."""
    return [
        [{"role": "assistant", "content": t1}],
        [{"role": "assistant", "content": t2}],
    ]


_ANSWER_JSON_OK = '{"学生得分": 10, "结论批改": "正确"}'
_ANSWER_JSON_BAD = '{"学生得分": 0, "结论批改": "错误"}'

WELL_FORMED = (
    "<think>"
    "已知 x=3，由题意得 y=2x=6。"
    "因此 y=6，代入验证 2x=6 成立。"
    "故答案为 y=6。"
    "</think>"
    "<answer>" + _ANSWER_JSON_OK + "</answer>"
)

EMPTY_THINK = "<think></think><answer>" + _ANSWER_JSON_BAD + "</answer>"

NO_TAGS = "the answer is 6"

SOLUTION_CORRECT = '{"学生得分": 10, "结论批改": "正确"}'
SOLUTION_WRONG = '{"学生得分": 0, "结论批改": "错误"}'


def check_topo_reward():
    tr = TopoReward()
    r_good = tr(_wrap(WELL_FORMED))[0]
    r_empty = tr(_wrap(EMPTY_THINK))[0]
    r_none = tr(_wrap(NO_TAGS))[0]

    assert r_good >= r_empty, (
        "well-formed topo ({:.4f}) should >= empty ({:.4f})".format(r_good, r_empty)
    )
    assert r_empty >= r_none, (
        "empty topo ({:.4f}) should >= no-tags ({:.4f})".format(r_empty, r_none)
    )
    assert 0.0 <= r_good <= 1.0
    print("  [PASS] TopoReward: good={:.4f} empty={:.4f} none={:.4f}".format(
        r_good, r_empty, r_none))


def check_continuity_reward():
    cr = ContinuityReward()
    r_good = cr(_wrap(WELL_FORMED))[0]
    r_empty = cr(_wrap(EMPTY_THINK))[0]
    assert 0.0 <= r_good <= 1.0
    assert r_good >= r_empty
    print("  [PASS] ContinuityReward: good={:.4f} empty={:.4f}".format(r_good, r_empty))


def check_format_reward():
    fr = FormatReward()
    r_good = fr(_wrap(WELL_FORMED))[0]
    r_none = fr(_wrap(NO_TAGS))[0]
    assert r_good > r_none, (
        "well-formed format ({:.4f}) should > no-tags ({:.4f})".format(r_good, r_none)
    )
    print("  [PASS] FormatReward: good={:.4f} none={:.4f}".format(r_good, r_none))


def check_outcome_reward():
    orw = OutcomeReward()
    r_match = orw(
        _wrap(WELL_FORMED),
        solution=[SOLUTION_CORRECT],
    )[0]
    r_mismatch = orw(
        _wrap(WELL_FORMED),
        solution=[SOLUTION_WRONG],
    )[0]
    assert r_match >= r_mismatch, (
        "match ({:.4f}) should >= mismatch ({:.4f})".format(r_match, r_mismatch)
    )
    assert r_match > 0.0, "match reward should be > 0"
    print("  [PASS] OutcomeReward: match={:.4f} mismatch={:.4f}".format(r_match, r_mismatch))


def check_hierarchical_ordering():
    hr = TopoHierarchicalReward()
    completions = _wrap2(WELL_FORMED, EMPTY_THINK)
    solutions = [SOLUTION_CORRECT, SOLUTION_WRONG]
    rewards = hr(completions, solution=solutions, reference_dag=[None, None])
    assert len(rewards) == 2
    assert rewards[0] >= rewards[1], (
        "correct ({:.4f}) should >= wrong ({:.4f})".format(rewards[0], rewards[1])
    )
    print("  [PASS] TopoHierarchicalReward: correct={:.4f} wrong={:.4f}".format(
        rewards[0], rewards[1]))


def check_composite_range():
    cr = TopoCompositeReward()
    rewards = cr(
        _wrap(WELL_FORMED),
        solution=[SOLUTION_CORRECT],
        reference_dag=[None],
    )
    assert all(0.0 <= r <= 1.0 for r in rewards)
    assert rewards[0] > 0.0, "composite reward should be > 0 for well-formed input"
    print("  [PASS] TopoCompositeReward: reward={:.4f} in [0,1]".format(rewards[0]))


def main():
    print("=" * 60)
    print("  TopoPRM Reward Invariant Check")
    print("=" * 60)

    checks = [
        ("TopoReward", check_topo_reward),
        ("ContinuityReward", check_continuity_reward),
        ("FormatReward", check_format_reward),
        ("OutcomeReward", check_outcome_reward),
        ("TopoHierarchicalReward", check_hierarchical_ordering),
        ("TopoCompositeReward", check_composite_range),
    ]

    passed = 0
    failed = 0
    for name, fn in checks:
        try:
            fn()
            passed += 1
        except AssertionError as e:
            print("  [FAIL] {}: {}".format(name, e))
            failed += 1
        except Exception as e:
            print("  [ERROR] {}: {}".format(name, e))
            failed += 1

    print("\nResults: {} passed, {} failed out of {}".format(passed, failed, len(checks)))
    if failed:
        sys.exit(1)
    print("All invariants hold. Safe to proceed.")


if __name__ == "__main__":
    main()
