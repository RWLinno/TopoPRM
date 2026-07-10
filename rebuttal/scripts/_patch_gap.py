#!/usr/bin/env python3
from pathlib import Path

f = Path("rebuttal/response.md")
t = f.read_text()

# Fill the B5w7 gap table placeholders
repl = {
    "[[GAP:math_wrong_hi]]": "0.79 (n=19)",
    "[[GAP:math_corr_lo]]": "n/a*",
    "[[GAP:gsm_wrong_hi]]": "0.64 (n=11)",
    "[[GAP:gsm_corr_lo]]": "n/a*",
}
for k, v in repl.items():
    t = t.replace(k, v)

# Add an explanatory note after the B5w7-1 table if not present.
note = ("\n\n*Almost all base-model traces receive a high structural score "
        "(mean q_topo=0.82), so the low-topology cell is nearly empty: the "
        "extractor finds surface structure even in wrong traces. That is "
        "precisely the point \u2014 across 137 held-out traces, **Pr(wrong | "
        "q_topo>0.8)=0.73** (0.64 GSM8K, 0.79 MATH-500), so a high topology "
        "score is frequently attached to an incorrect answer. This is exactly "
        "why correctness must remain the primary, non-overridable gate "
        "(multiplicative reward + ACE), and why we scope TopoPRM as a "
        "process signal rather than a correctness proxy.")
anchor = "| GSM8K | [[GAP:gsm_wrong_hi]]"  # will already be replaced; use post-table anchor
# insert note after the B5w7-1 table block (before "This explains why")
marker = "This explains why TopoPRM improves average accuracy"
if marker in t and "Almost all base-model traces" not in t:
    t = t.replace(marker, note.strip() + "\n\n" + marker)

f.write_text(t)
print("patched semantic gap numbers")
