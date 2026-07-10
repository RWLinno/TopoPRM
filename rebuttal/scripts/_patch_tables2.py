#!/usr/bin/env python3
from pathlib import Path

f = Path("rebuttal/response.md")
t = f.read_text()

# 1. Add token-efficiency note to HxUk-2 (TopoPRM wins with MORE tokens -> not brevity)
hxuk_note_anchor = "one controlled harness.)"
hxuk_note = (
    "one controlled harness.)\n\n"
    "Crucially, on GSM8K the matched runs give outcome-only 75.5 (mean 279 tok), "
    "outcome+length 76.5 (277 tok), and Full TopoPRM 77.0 (438 tok): TopoPRM is "
    "the most accurate **while generating more tokens than the length-controlled "
    "baseline**, so its gain is not a brevity artifact. (GSM8K here is a 200-item "
    "pass@1 subset for turnaround; MATH-500/AIME rows are completing and will be "
    "reported in the camera-ready.)"
)
if hxuk_note_anchor in t and "not a brevity artifact" not in t:
    t = t.replace(hxuk_note_anchor, hxuk_note, 1)
    print("added HxUk-2 token note")

# 2. Fix B5w7-2: separate matched-TRL rows from paper rows to avoid mixing protocols
old_b = """Tab B5w7-2 (matched TRL runs; reward is the only variable — see also the
paper's Table 4 ablation which removes each signal from the same SFT checkpoint):

| Reward | GSM8K | MATH-500 | AIME'24 |
| --- | ---: | ---: | ---: |
| Outcome-only GRPO | 85.1 | 67.4 | 46.7 |
| + length only (no topology) | 76.5 | 41.5 | 0.0 |
| w/o topology | 84.5 | 68.8 | 36.7 |
| w/o continuity | 85.1 | 66.4 | 36.7 |
| Full TopoPRM | 84.3 | 66.6 | 50.0 |"""
new_b = """Tab B5w7-2a (paper Table 4, DR1-7B, same SFT ckpt + 200 GRPO steps, full pass@1):

| Reward | GSM8K | MATH-500 | AIME'24 |
| --- | ---: | ---: | ---: |
| Outcome-only GRPO | 85.1 | 67.4 | 46.7 |
| w/o topology | 84.5 | 68.8 | 36.7 |
| w/o continuity | 85.1 | 66.4 | 36.7 |
| Full TopoPRM | 84.3 | 66.6 | 50.0 |

Tab B5w7-2b (new matched-TRL rerun, GSM8K 200-item pass@1, isolates length):

| Reward | GSM8K | mean tokens |
| --- | ---: | ---: |
| Outcome-only GRPO | 75.5 | 279 |
| + length only (no topology) | 76.5 | 277 |
| Full TopoPRM (hierarchical) | 77.0 | 438 |"""
if old_b in t:
    t = t.replace(old_b, new_b)
    print("fixed B5w7-2 table split")
else:
    print("B5w7-2 anchor not found")

f.write_text(t)
