#!/usr/bin/env python3
"""One-off: enrich the HxUk W1 paragraph with the edge-type reliability table."""
from pathlib import Path

f = Path("rebuttal/response.md")
text = f.read_text()

old = (
    "Against these labels, the extractor's edges score **P=0.48, R=0.59, F1=0.53**. "
    "Reliability is edge-type dependent: expression/claim-reference edges are most "
    "precise (0.59-0.63 precision), while variable-only and fallback sequential edges "
    "are weaker (0.25 precision). We will add this as an independent diagnostic "
    "(Appendix E) and revise the claim from \"logical dependencies\" to "
    "\"surface-evidenced support dependencies.\" Our claim is not proof-graph recovery; "
    "it is that these edges carry process-supervision signal that outcome-only rewards "
    "cannot see."
)
new = (
    "Against these labels the extractor scores **P=0.48, R=0.59, F1=0.53** over all "
    "edge types, with strongly edge-type-dependent reliability:\n\n"
    "Tab HxUk-EV (edge-type precision vs. independent judge, 120 traces):\n\n"
    "| Edge type | Precision | TP | FP |\n"
    "| --- | ---: | ---: | ---: |\n"
    "| expression-overlap | 0.63 | 19 | 11 |\n"
    "| implicit-block | 0.63 | 15 | 9 |\n"
    "| order (fallback seq.) | 0.61 | 127 | 82 |\n"
    "| expression-ref | 0.59 | 49 | 34 |\n"
    "| variable-ref | 0.25 | 49 | 145 |\n\n"
    "Expression/claim/order edges are reliable (0.59-0.63), while **variable-only edges "
    "are the dominant error source: precision 0.25, accounting for 52% of all false "
    "positives** \u2014 exactly the \"variable overlap\" failure mode the reviews "
    "anticipated. Excluding var-ref edges raises precision to **0.61**. We will "
    "(i) down-weight/gate variable-only edges in the extractor, (ii) report this as an "
    "independent diagnostic (Appendix E), and (iii) revise the claim from \"logical "
    "dependencies\" to \"surface-evidenced support dependencies.\" This is not "
    "proof-graph recovery; it confirms the edges carry process-supervision signal "
    "outcome-only rewards cannot see, and localizes where the extractor is weakest."
)

if old in text:
    text = text.replace(old, new)
    f.write_text(text)
    print("patched W1")
else:
    print("OLD NOT FOUND (already patched or text drift)")
