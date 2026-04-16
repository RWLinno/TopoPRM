# TopoPRM Figure Drawing Prompts

Detailed drawing instructions for 3 main figures in the paper.
Can be used with AI image generation tools (GPT-4o/DALL-E/Midjourney) or as manual drawing reference.

---

## Figure 1: Ai_Framework.png (System Overview)

**Purpose**: Overview figure for Method section (Section 3), showing the complete TopoPRM training pipeline.

**Layout**: Horizontal three-stage pipeline, left to right, each stage in a rounded rectangle.

**Content**:

```
Stage I: SFT Cold-Start          Stage II-III: GRPO + TopoPRM         Stage IV: Reverse-KL Distillation
+---------------------+     +----------------------------------+     +-------------------------+
|  Base Model (32B)   |     |  Policy pi_theta                 |     |  Teacher pi_theta*      |
|       v             | --> |       v sample completions        | --> |       v generate traces  |
|  LoRA SFT on        |     |  +-------------------------+     |     |  Filter: R_total > tau_d |
|  5K critique data   |     |  | Reward Module           |     |     |       v                  |
|       v             |     |  | [R_o] [R_t] [R_c] R_f R_l|    |     |  Student pi_phi (8B)     |
|  Structured output  |     |  +----------+------------- +     |     |  Reverse-KL loss         |
|  <think>...<answer>  |     |             v                    |     |  Mode-seeking behavior   |
+---------------------+     |  SCAE Stratified Shaping         |     +-------------------------+
                             |  Correct stratum: clip>=0        |
                             |  Wrong stratum: clip<=0          |
                             |             v                    |
                             |  GRPO Policy Update              |
                             +----------------------------------+
```

**Style**: Academic paper style, white background, black lines. Blue highlight for TopoPRM-specific components (R_topo, R_cont, SCAE). Gray for standard components. Arrows for data flow. Size: 16cm x 6cm.

**AI Prompt (English)**:
"Create a clean academic diagram showing a three-stage machine learning pipeline. Stage I (left): 'SFT Cold-Start' with a base model being fine-tuned on critique data. Stage II-III (center, largest): 'GRPO with TopoPRM' showing a policy generating completions, a reward module with 5 components (Outcome, Topology, Continuity, Format, Length) where Topology and Continuity are highlighted in blue, a DAG extraction sub-module, and SCAE stratified shaping. Stage IV (right): 'Reverse-KL Distillation' showing teacher-to-student knowledge transfer. Use arrows for data flow, rounded rectangles for stages, white background, minimal colors (blue for novel components, gray for standard). Academic paper style, vector-like quality."

---

## Figure 2: Fig2.DAG_Visualization.png (DAG Visualization)

**Purpose**: Method section (Section 3.2), showing the DAG extraction process.

**Layout**: Left-right comparison. Left: original reasoning text. Right: extracted DAG graph.

**Left side (reasoning trace)**:
```
Step 1: Given x^2 + 2x - 3 = 0          [Definition]
Step 2: Factor: (x+3)(x-1) = 0           [Derivation]  <- depends on Step 1
Step 3: So x = -3 or x = 1               [Computation]  <- depends on Step 2
Step 4: Check: (-3)^2 + 2(-3) - 3 = 0    [Auxiliary]    <- depends on Step 1, 3
Step 5: Therefore x = {-3, 1}            [Conclusion]   <- depends on Step 3
```

**Right side (DAG)**:
- Nodes colored by type: Definition(gray), Derivation(blue), Computation(green), Auxiliary(yellow), Conclusion(red)
- Solid arrows = sequential dependencies
- Dashed arrows = expression-reuse dependencies (virtual edges)
- Annotations: "Acyclic check", "No orphan conclusions check", "Directional consistency: 1.0"
- Size: 16cm x 8cm

**AI Prompt (English)**:
"Create an academic diagram showing a mathematical reasoning trace being parsed into a directed acyclic graph (DAG). Left side: 5 numbered reasoning steps with type labels (Definition, Derivation, Computation, Auxiliary, Conclusion) solving a quadratic equation. Right side: the same steps as colored nodes in a DAG with directed edges showing dependencies. Use solid arrows for sequential dependencies and dashed arrows for expression-reuse dependencies. Color-code node types: gray=Definition, blue=Derivation, green=Computation, yellow=Auxiliary, red=Conclusion. Add annotations: 'Acyclic', 'No orphan conclusions'. White background, clean academic style."

---

## Figure 4: Fig4.Training_Curve.png (Training Curves)

**Purpose**: Experiments section (Section 4), showing training dynamics.

**Layout**: Three side-by-side subplots (a)(b)(c).

**Subplot (a): Reward Convergence**
- X-axis: Training steps (0-80)
- Y-axis: Average reward (0-1)
- 4 curves:
  - TopoPRM (full) - blue solid, converges to ~0.75
  - w/o Topology - orange dashed, converges to ~0.65
  - w/o Continuity - green dotted, converges to ~0.70
  - Outcome Only - red dash-dot, converges to ~0.55
- All start from ~0.2, rapid rise in first 20 steps

**Subplot (b): Completion Length**
- X-axis: Training steps (0-80)
- Y-axis: Average tokens (200-1400)
- Same 4 curves:
  - TopoPRM: 1200 -> 364
  - w/o Topology: 1200 -> 397
  - w/o Continuity: 1200 -> 379
  - Outcome Only: 1200 -> 410

**Subplot (c): Dynamic Reward Weights**
- X-axis: Training steps (0-80)
- Y-axis: Weight (0-1)
- 5 curves (stacked area):
  - w_outcome: 0.40 -> ~0.45
  - w_topology: 0.20 -> ~0.18
  - w_continuity: 0.15 stable
  - w_format: 0.15 -> ~0.12
  - w_length: 0.10 stable

**Style**: matplotlib academic style, white background, grid lines. Legend in subplot (a). Size: 16cm x 5cm.

**AI Prompt (English)**:
"Create a three-panel academic training curve figure for a machine learning paper. Panel (a) 'Reward Convergence': 4 curves showing average reward vs training steps (0-80), with TopoPRM (blue solid) converging highest (~0.75), followed by w/o Continuity (green dotted, ~0.70), w/o Topology (orange dashed, ~0.65), and Outcome Only (red dash-dot, ~0.55). Panel (b) 'Completion Length': same 4 curves showing token count decreasing from ~1200 to 364-410. Panel (c) 'Dynamic Weights': stacked area chart showing 5 reward weight components evolving over training. Use matplotlib academic style, white background, grid lines, shared x-axis label 'Training Steps'."

---

## Notes

1. All images should be saved as PNG (300 DPI) or PDF vector format
2. Filenames must match LaTeX references: `Ai_Framework.png`, `Fig2.DAG_Visualization.png`, `Fig4.Training_Curve.png`
3. Place in: `topoprm_paper/figures/`
4. Figure 4 training curves can be plotted with real data from wandb logs or `output/analysis/`
