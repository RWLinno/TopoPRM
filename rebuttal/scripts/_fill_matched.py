#!/usr/bin/env python3
"""Fill [[MATCH:*]] placeholders in response.md from matched eval metrics."""
import glob
import json
import os
from pathlib import Path

def get(label, bench):
    fp = f"rebuttal/outputs/eval_tables/{label}_matched_{bench}_metrics.json"
    if os.path.exists(fp):
        d = json.load(open(fp))
        return f"{d['pass@1']*100:.1f}"
    return None  # keep placeholder until the metric exists

# variant -> placeholder prefix
V = {"topo_hier": "th", "outcome_only": "oo", "outcome_length": "ol"}
B = {"gsm8k": "gsm", "math500": "math", "aime2024": "aime"}

f = Path("rebuttal/response.md")
t = f.read_text()
for label, pre in V.items():
    for bench, bs in B.items():
        val = get(label, bench)
        if val is not None:
            t = t.replace(f"[[MATCH:{pre}_{bs}]]", val)
f.write_text(t)

# report what remains
import re
left = sorted(set(re.findall(r"\[\[MATCH:[^\]]+\]\]", t)))
print("filled. remaining MATCH placeholders:", left)
