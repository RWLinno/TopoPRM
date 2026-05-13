# Figure Generation Prompts for TopoPRM Paper

## Figure 1: Method Overview (Main figure, full-width)

**Prompt for diagram tool / tikz / designer:**

Create a clean, horizontal three-stage pipeline diagram for an NLP paper:

Stage I (left): "Reward Module" 
- Show a reasoning trace being parsed into a DAG (small graph with ~5 nodes and directed edges)
- Below the DAG: three reward signals branch out: R_topo (topological structure), R_cont (continuity), R_out (outcome)
- These feed into a "Hierarchical Aggregation" box producing R_total

Stage II (middle): "Post-Training (SFT → GRPO)"
- Show the R_total signal feeding into a GRPO training loop
- Small icon of a model being updated
- Label: "Teacher (7B)"

Stage III (right): "TVSD Compression"  
- Teacher generates on-policy rollouts
- DAG diagnostics filter/densify supervision
- Arrow to smaller "Student (≤4B)" model
- Label the on-policy loop

Style: Flat, minimal, blue/gray color scheme. No gradients. Use thin arrows. Similar to figures in ICLR 2024/NeurIPS 2024 papers. Width = full textwidth.

---

## Figure 2: DAG Example (Column figure)

**Prompt:**

Show a concrete reasoning DAG for a simple math problem (e.g., "Solve 3x + 5 = 20"):
- 4-5 nodes representing reasoning steps
- Directed edges showing dependencies (with small labels like "substitution", "arithmetic")
- One node highlighted in red as "orphan conclusion" (no incoming edge from valid predecessors)
- One dashed edge marked as "virtual edge" (implied but not explicitly stated dependency)
- Color scheme: valid nodes in light blue, orphan in light red, edges in gray, virtual edges dashed

---

## Figure 3: Reward Landscape / Ablation Radar (Half-width)

**Prompt:**

Create a radar/spider chart with 5 axes:
- GSM8K, MATH-500, AIME 2024, MMLU, Avg Tokens (inverted, lower is better)

Plot 4 lines:
- Baseline (gray, dashed)
- + Outcome-only (orange)
- + TopoPRM full (blue, bold)
- + TVSD Student 4B (green, dotted)

Use pass@1 values from Table 1 (use approximate values: baseline ~83/56/23/42, outcome-only ~87/63/30/43, full ~89/65/33/44, student ~85/60/27/43).

---

## Figure 4: Training Dynamics (Half-width)

**Prompt:**

Line plot showing training steps (x-axis, 0-200) vs. reward (y-axis):
- 4 curves: outcome-only, no-topo, no-continuity, TopoPRM-full
- TopoPRM-full should show highest final reward and steepest initial climb
- outcome-only lowest
- Include a secondary y-axis or inset for "reward std" (to show collapse avoidance)

This will be generated from actual wandb data after training completes.

---

## Figure 5: DAG Structural Quality (Half-width)

**Prompt:**

Grouped bar chart:
- X-axis: 3 models (Baseline, SFT, TopoPRM)
- Y-axis left: percentage (0-100%)
- 3 grouped bars per model: Acyclic%, No-Orphan%, Edge-Keep%
- TopoPRM should show highest on all three metrics
- Color: light blue, medium blue, dark blue for the three metrics

---

## Figure 6: Accuracy-Tokens Pareto Frontier (Half-width)

**Prompt:**

Scatter plot:
- X-axis: Average output tokens (log scale or linear, range 200-4000)
- Y-axis: MATH-500 pass@1 (range 50-70%)
- Points: Baseline, SFT, TopoPRM-7B, TVSD-4B, SFT-distill-4B
- TopoPRM-7B and TVSD-4B should be on the Pareto frontier
- Draw a dashed Pareto frontier line
- Annotate each point with model name

---

## Style Guidelines

- Font: matches LaTeX (Computer Modern or similar serif)
- Colors: use a colorblind-friendly palette (blue, orange, green, red, purple)
- Resolution: 300 DPI minimum
- Format: PDF vector preferred for LaTeX inclusion
- Width: full-width figures = 6.5in, half-width = 3.1in
