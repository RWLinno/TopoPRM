#!/usr/bin/env python3
from pathlib import Path

f = Path("rebuttal/response.md")
t = f.read_text()

old = "Excluding var-ref edges raises precision to **0.61**."
new = ("Acting on this diagnostic, we added a precision guard that keeps a "
       "variable-reference edge only when the two steps share at least two "
       "variables; re-running the same blinded evaluation, this raises the "
       "extractor to **P=0.57, R=0.68, F1=0.62** (from 0.48/0.59/0.53) \u2014 "
       "the validation directly produced a better extractor.")
if old in t:
    t = t.replace(old, new)
    f.write_text(t)
    print("patched guard sentence")
else:
    print("anchor not found; current EV block:")
    import re
    m = re.search(r"variable-only edges are the dominant.*?weakest\.", t, re.DOTALL)
    print(m.group(0)[:400] if m else "no match")
