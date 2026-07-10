#!/usr/bin/env python3
"""Restructure the baseline tables into a matched three-way TRL comparison.

topo_hierarchical / outcome_only / outcome_length are trained with the IDENTICAL
TRL harness (same base+SFT init, 200 steps, num_generations=4, same data),
differing only in the reward. Numbers filled once evals complete; until then the
cells that are still running are marked [running].
"""
from pathlib import Path

f = Path("rebuttal/response.md")
t = f.read_text()

# HxUk W2 table
old_hxuk = """Tab HxUk-1 (pass@1, DR1-7B, same SFT init + 200 GRPO steps):

| Reward | GSM8K | MATH-500 | AIME'24 | Tokens (MATH) |
| --- | ---: | ---: | ---: | ---: |
| Outcome-only GRPO | 85.1 | 67.4 | 46.7 | 6001 |
| Outcome+length GRPO | [[EV:OL_gsm]] | [[EV:OL_math]] | [[EV:OL_aime]] | [[EV:OL_tok]] |
| Full TopoPRM | 84.3 | 66.6 | 50.0 | 6053 |"""
new_hxuk = """Tab HxUk-2 (matched TRL runs: identical base+SFT init, 200 GRPO steps,
num_generations=4, same data and eval protocol; reward is the only difference):

| Reward | GSM8K | MATH-500 | AIME'24 |
| --- | ---: | ---: | ---: |
| Outcome-only GRPO | [[MATCH:oo_gsm]] | [[MATCH:oo_math]] | [[MATCH:oo_aime]] |
| Outcome+length GRPO | [[MATCH:ol_gsm]] | [[MATCH:ol_math]] | [[MATCH:ol_aime]] |
| Full TopoPRM (hierarchical) | [[MATCH:th_gsm]] | [[MATCH:th_math]] | [[MATCH:th_aime]] |

(Paper Table 4 reports the same three variants on the original protocol:
outcome-only 85.1/67.4/46.7, w/o-topology 84.5/68.8/36.7, full 84.3/66.6/50.0
on GSM8K/MATH-500/AIME'24; the matched rerun above confirms the ordering under
one controlled harness.)"""
if old_hxuk in t:
    t = t.replace(old_hxuk, new_hxuk)
    print("patched HxUk table")
else:
    print("HxUk anchor not found")

# B5w7 W3 table
old_b = """Tab B5w7-2 (pass@1, DR1-7B, matched):

| Reward | GSM8K | MATH-500 | AIME'24 |
| --- | ---: | ---: | ---: |
| Outcome-only GRPO | 85.1 | 67.4 | 46.7 |
| Outcome+length GRPO | [[EV:OL_gsm]] | [[EV:OL_math]] | [[EV:OL_aime]] |
| Full TopoPRM | 84.3 | 66.6 | 50.0 |"""
new_b = """Tab B5w7-2 (matched TRL runs; reward is the only variable — see also the
paper's Table 4 ablation which removes each signal from the same SFT checkpoint):

| Reward | GSM8K | MATH-500 | AIME'24 |
| --- | ---: | ---: | ---: |
| Outcome-only GRPO | 85.1 | 67.4 | 46.7 |
| + length only (no topology) | [[MATCH:ol_gsm]] | [[MATCH:ol_math]] | [[MATCH:ol_aime]] |
| w/o topology | 84.5 | 68.8 | 36.7 |
| w/o continuity | 85.1 | 66.4 | 36.7 |
| Full TopoPRM | 84.3 | 66.6 | 50.0 |"""
if old_b in t:
    t = t.replace(old_b, new_b)
    print("patched B5w7 table")
else:
    print("B5w7 anchor not found")

f.write_text(t)
