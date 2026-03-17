# Figure Generation Prompts

Below are the prompts for generating each figure in the paper.
Use these with an AI image generation tool (e.g., GPT-4o, Midjourney)
or drawing tools (draw.io, Figma, TikZ).

---

## Figure 1: TopoPRM Framework Overview

```
Prompt for academic architecture diagram:

Draw a professional NeurIPS-style framework overview diagram for "TopoPRM" with a clean white background and flat vector style.

The diagram has three stages flowing left to right:

Stage 1 (SFT): A box labeled "Qwen3-32B + LoRA" with an arrow from "Math Reasoning Data (5K samples)" entering it. Output arrow labeled "SFT Model".

Stage 2 (GRPO with TopoPRM): This is the main section, larger than others.
- Top: "SFT Model" feeds into a "Policy π_θ" block
- Policy generates multiple completions (show 8 branching arrows)
- Each completion goes through a "Reward Computation" module containing 5 stacked components:
  * Outcome Reward (green, weight 0.4)
  * Topological Structure Reward (blue, weight 0.2) - show a small DAG icon
  * Continuity Reward (orange, weight 0.15)
  * Format Reward (gray, weight 0.15)
  * Length Reward (gray, weight 0.1)
- Rewards feed into "SCAE" (Stratified Clipping Advantage Estimation) block
- SCAE shows two groups: "Correct ✓" (green) and "Wrong ✗" (red) with asymmetric clipping
- SCAE output feeds back to update π_θ via "GRPO Loss"

Stage 3 (Distillation): "32B Teacher" arrow to "Reverse-KL" to "14B/7B Student"

Color scheme: Use soft pastels - light blue for model blocks, light green for correct/outcome, light red for wrong, light orange for topology, white background. No gradients. Thin black borders.

Style: DeepMind/OpenAI paper figure aesthetic. Minimalist, professional, no 3D effects.
```

---

## Figure 2: DAG Extraction Example

```
Prompt for DAG visualization:

Create an academic figure showing how a mathematical reasoning trace is converted into a DAG (Directed Acyclic Graph).

Left side: A text box showing a math reasoning trace with 6 numbered steps:
Step 1: "Given: x + y = 10" (labeled: DEFINITION, blue)
Step 2: "Given: 2x - y = 5" (labeled: DEFINITION, blue)
Step 3: "Adding equations: 3x = 15" (labeled: COMPUTATION, green)
Step 4: "Therefore x = 5" (labeled: DERIVATION, orange)
Step 5: "Substituting: y = 10 - 5 = 5" (labeled: SUBSTITUTION, purple)
Step 6: "Conclusion: x = 5, y = 5" (labeled: CONCLUSION, red)

Right side: The corresponding DAG with:
- Nodes as rounded rectangles colored by type
- Directed edges showing dependencies:
  Step 1 → Step 3, Step 1 → Step 5
  Step 2 → Step 3
  Step 3 → Step 4
  Step 4 → Step 5
  Step 4 → Step 6, Step 5 → Step 6
- Edge labels: "dependency" (solid) vs "sequential" (dashed)

Below the DAG: Structural metrics checkboxes:
✓ Acyclic  ✓ Connected  ✓ Directionally consistent  ✓ No orphan conclusions

White background, clean academic style matching NeurIPS format.
```

---

## Figure 3: SCAE Advantage Computation

```
Prompt for SCAE visualization:

Create a figure comparing standard GRPO advantage vs SCAE advantage computation.

Two panels side by side:

Left panel "Standard GRPO":
- A group of 8 completions shown as horizontal bars
- Each bar colored by reward (gradient from red=low to green=high)
- Advantages computed as: A_i = (r_i - mean(r)) / std(r)
- Problem highlighted: a wrong answer with high structural reward gets positive advantage (marked with ⚠️)

Right panel "SCAE (Ours)":
- Same 8 completions, but now partitioned into two groups:
  * Top group "Correct (C)" in green background: 3 completions
  * Bottom group "Wrong (W)" in red background: 5 completions
- Within Correct group: only upward auxiliary bonus (structural reward above group mean)
- Within Wrong group: only downward auxiliary penalty
- Key property highlighted: "Correct always ≥ 0, Wrong always ≤ 0"

Clean academic style, soft colors, white background. Arrows and annotations explaining the key difference.
```

---

## Figure 4: Training Dynamics

```
Prompt for training curves:

Create a 2x1 subplot figure showing GRPO training dynamics.

Top plot "Reward vs Training Steps":
- X-axis: Training steps (0 to 200)
- Y-axis: Composite Reward (0 to 0.1)
- Lines for: TopoPRM (full, blue), Outcome Only (green), w/o Topo (orange), w/o Continuity (purple)
- TopoPRM converges fastest and highest
- Show checkpoint markers at step 50, 100, 150

Bottom plot "Mean Completion Length vs Training Steps":
- X-axis: Training steps (0 to 200)
- Y-axis: Mean length in characters (500 to 2000)
- Same color scheme
- TopoPRM shows steeper length reduction (more concise reasoning)

Grid lines, legend in upper right. Clean matplotlib academic style.
Font: serif, matching LaTeX Computer Modern.
```
