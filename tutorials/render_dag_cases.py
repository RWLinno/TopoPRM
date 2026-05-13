"""Render real reasoning DAGs from the TopoPRM training data.

Usage
-----
    python tutorials/render_dag_cases.py              # 8 mixed cases
    python tutorials/render_dag_cases.py --n 10 --source math
    python tutorials/render_dag_cases.py --from-rollout <jsonl>   # use model output

This script bypasses the Streamlit GUI in ``src/gui/dag_reward_viewer.py`` and
uses the same :class:`ReasoningDAG` engine directly so we can render a batch of
cases into static images without a browser.

Output
------
    topoprm_paper/figures/dag_cases/case_<id>.png   — one per case
    topoprm_paper/figures/dag_cases/pack.pdf         — all cases on one sheet
    topoprm_paper/figures/dag_cases/summary.md       — Q/A/metrics per case

Notes
-----
- By default we render the *reference* DAGs stored in
  ``data/grpo_ready/train_public.jsonl`` (field ``reference_dag``). These are
  deterministic DAGs extracted from the ground-truth solutions by the same
  rule-based pipeline described in the paper, so they are real data, not
  synthetic.
- To render DAGs over **model-generated** traces instead, pass
  ``--from-rollout`` with a JSONL that has a ``y_init`` or ``response`` field
  containing the full ``<think>...</think>`` text. Produce such a file by
  running ``scripts/bench_transformers.py`` with ``--save_solutions`` (see the
  AIME24 command printed at the bottom of this module).
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import random
import re
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

# Allow `python3 tutorials/render_dag_cases.py` from the repo root without
# exporting PYTHONPATH explicitly.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx

from src.dag.graph import (
    DOUBLE_BARRIER_EDGE,
    SOLID_EDGE,
    VIRTUAL_EDGE,
    ReasoningDAG,
)
from src.dag.node import StepType
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_JSONL = REPO_ROOT / "data" / "grpo_ready" / "train_public.jsonl"
DEFAULT_OUT_DIR = REPO_ROOT / "topoprm_paper" / "figures" / "dag_cases"


PALETTE = {
    "solid": "#2E6FB5",
    "virtual": "#D97706",
    "barrier": "#9CA3AF",
    "orphan": "#C0392B",
    "node": "#2E6FB5",
    "node_orphan": "#C0392B",
    "text": "#333333",
    "panel_bg": "#F5F5F5",
}

NODE_SHAPE = {
    StepType.DEFINITION: "o",
    StepType.DERIVATION: "s",
    StepType.COMPUTATION: "s",
    StepType.CONCLUSION: "H",
    StepType.AUXILIARY: "o",
    StepType.SUBSTITUTION: "s",
    StepType.CASE_ANALYSIS: "D",
    StepType.UNKNOWN: "s",
}


@dataclass
class Case:
    case_id: str
    source: str
    question: str
    answer: str
    dag: ReasoningDAG
    raw_steps: List[str]
    topo: Optional[float] = None
    continuity: Optional[float] = None


def _load_reference_dag(row: Dict[str, Any]) -> Optional[ReasoningDAG]:
    raw = row.get("reference_dag")
    if raw is None:
        return None
    if isinstance(raw, str):
        try:
            raw = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            raw = json.loads(raw)
    try:
        return ReasoningDAG.from_dict(raw)
    except Exception:
        return None


def _raw_steps_from_reference(raw_dag: Dict[str, Any]) -> List[str]:
    steps = []
    for n in raw_dag.get("nodes", []):
        steps.append(str(n.get("raw_text", "")).strip())
    return steps


def _coerce_reference_dict(value: Any) -> Dict[str, Any]:
    if isinstance(value, str):
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return json.loads(value)
    return value


def _completion_from_steps(raw_steps: Sequence[str]) -> list:
    joined = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(raw_steps))
    content = f"<think>\n{joined}\n</think>\n<answer>placeholder</answer>"
    return [{"role": "assistant", "content": content}]


def _structural_scores(case: "Case") -> tuple[float, float]:
    """Structural approximations of q_topo / q_cont.

    The production rewards in ``src/reward`` depend on the training framework
    (swift/trl) which is not installed in every environment. We therefore
    compute a deterministic structural approximation that mirrors the four
    terms of q_topo described in the paper: acyclicity, no-orphan, direction
    consistency, and step alignment.
    """
    g = case.dag.graph
    n_nodes = g.number_of_nodes()
    n_edges = g.number_of_edges()
    if n_nodes == 0:
        return 0.0, 0.0

    try:
        acyclic = 1.0 if nx.is_directed_acyclic_graph(g) else 0.0
    except Exception:
        acyclic = 0.0

    orphans = set(case.dag.orphan_nodes())
    no_orphan = 1.0 - len(orphans) / float(n_nodes)

    forward = sum(1 for u, v in g.edges() if u < v)
    direction = forward / float(n_edges) if n_edges else 1.0

    layered = 0
    try:
        for layer in nx.topological_generations(g):
            if len(layer) > 0:
                layered += len(layer)
    except nx.NetworkXUnfeasible:
        layered = 0
    step_align = layered / float(n_nodes) if n_nodes else 0.0

    q_topo = 0.3 * acyclic + 0.3 * no_orphan + 0.2 * direction + 0.2 * step_align

    # Continuity: fraction of adjacent step pairs connected by any edge.
    adj_pairs = 0
    for u in range(n_nodes - 1):
        if g.has_edge(u, u + 1) or g.has_edge(u + 1, u):
            adj_pairs += 1
    q_cont = adj_pairs / float(max(n_nodes - 1, 1))

    return float(min(max(q_topo, 0.0), 1.0)), float(min(max(q_cont, 0.0), 1.0))


def load_cases_from_training(
    jsonl_path: Path,
    n: int,
    source: str = "mixed",
    min_nodes: int = 4,
    max_nodes: int = 9,
    seed: int = 0,
    prefer_fresh_extractor: bool = True,
) -> List[Case]:
    """Stratified sample of reasoning chains.

    When ``prefer_fresh_extractor`` is True (default), we re-run
    ``parse_answer_to_dag_debug`` on each row's ``standard_answer`` so the
    rendered DAG reflects the *current* extractor, not the stale
    ``reference_dag`` blob that may have been cached months ago.
    """
    from src.data.build_dag import parse_answer_to_dag_debug

    rng = random.Random(seed)
    pool: List[Case] = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if source != "mixed" and row.get("source") != source:
                continue

            dag: Optional[ReasoningDAG] = None
            raw_steps: List[str] = []
            if prefer_fresh_extractor:
                text = (
                    row.get("standard_answer")
                    or row.get("solution")
                    or row.get("answer")
                    or ""
                )
                if text:
                    try:
                        dag, debug = parse_answer_to_dag_debug(
                            text, problem_id=row.get("record_id", "case")
                        )
                        raw_steps = [
                            s.get("normalized_text", s.get("raw_text", ""))
                            for s in debug.get("steps", [])
                        ]
                    except Exception:
                        dag = None
            if dag is None or dag.graph.number_of_nodes() == 0:
                dag = _load_reference_dag(row)
                if dag is None:
                    continue
                raw_dag = _coerce_reference_dict(row["reference_dag"])
                raw_steps = _raw_steps_from_reference(raw_dag)

            node_count = dag.graph.number_of_nodes()
            if not (min_nodes <= node_count <= max_nodes):
                continue
            pool.append(
                Case(
                    case_id=row.get("record_id", "unknown"),
                    source=row.get("source", "unknown"),
                    question=row.get("question", ""),
                    answer=str(row.get("final_answer", "")),
                    dag=dag,
                    raw_steps=raw_steps,
                )
            )
    if not pool:
        return []
    rng.shuffle(pool)

    picked: List[Case] = []
    seen_sources: Dict[str, int] = {}
    for case in pool:
        if len(picked) >= n:
            break
        cap = math.ceil(n * 0.7)
        if source == "mixed" and seen_sources.get(case.source, 0) >= cap:
            continue
        case.topo, case.continuity = _structural_scores(case)
        picked.append(case)
        seen_sources[case.source] = seen_sources.get(case.source, 0) + 1
    if len(picked) < n:
        for case in pool:
            if case in picked or len(picked) >= n:
                continue
            case.topo, case.continuity = _structural_scores(case)
            picked.append(case)
    return picked[:n]


def load_cases_from_rollout(
    jsonl_path: Path, n: int, seed: int = 0
) -> List[Case]:
    rng = random.Random(seed)
    rows: List[Dict[str, Any]] = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    rng.shuffle(rows)
    cases: List[Case] = []
    for row in rows:
        if len(cases) >= n:
            break
        text = (
            row.get("y_init")
            or row.get("response")
            or row.get("completion")
            or row.get("pred_raw")
            or ""
        )
        if not text:
            continue
        case_id = str(
            row.get("problem_id")
            or row.get("record_id")
            or row.get("question", "")[:32]
        )
        try:
            from src.data.build_dag import parse_answer_to_dag_debug
        except ImportError as e:
            raise SystemExit(
                "src.data.build_dag is unavailable in this environment; "
                "use the default reference-DAG mode instead."
            ) from e
        dag, debug = parse_answer_to_dag_debug(text, problem_id=case_id)
        if dag.graph.number_of_nodes() < 3:
            continue
        case = Case(
            case_id=case_id,
            source=row.get("source", "rollout"),
            question=row.get("question", row.get("problem", "")),
            answer=str(row.get("gold", row.get("final_answer", ""))),
            dag=dag,
            raw_steps=[s.get("normalized_text", s.get("raw_text", "")) for s in debug.get("steps", [])],
        )
        case.topo, case.continuity = _structural_scores(case)
        cases.append(case)
    return cases


def _layered_positions(dag: ReasoningDAG) -> Dict[int, tuple[float, float]]:
    g = dag.graph
    try:
        generations = list(nx.topological_generations(g))
    except nx.NetworkXUnfeasible:
        generations = [list(g.nodes())]
    pos: Dict[int, tuple[float, float]] = {}
    for layer_idx, layer in enumerate(generations):
        layer_sorted = sorted(layer)
        count = max(1, len(layer_sorted))
        for i, node_id in enumerate(layer_sorted):
            y = (i - (count - 1) / 2.0) * 1.0
            pos[node_id] = (float(layer_idx), float(y))
    return pos


def _edge_buckets(dag: ReasoningDAG):
    solid, virtual, barrier = [], [], []
    for u, v, d in dag.graph.edges(data=True):
        et = d.get("edge_type", "")
        if et == VIRTUAL_EDGE:
            virtual.append((u, v))
        elif et == DOUBLE_BARRIER_EDGE:
            barrier.append((u, v))
        else:
            solid.append((u, v))
    return solid, virtual, barrier


def render_case_axes(ax: plt.Axes, case: Case) -> None:
    dag = case.dag
    g = dag.graph
    pos = _layered_positions(dag)
    if not pos:
        ax.text(0.5, 0.5, "empty DAG", ha="center", va="center")
        ax.set_axis_off()
        return
    orphans = set(dag.orphan_nodes())

    node_colors = [
        PALETTE["node_orphan"] if n in orphans else PALETTE["node"]
        for n in g.nodes()
    ]
    node_edges = [
        PALETTE["orphan"] if n in orphans else PALETTE["text"]
        for n in g.nodes()
    ]
    nx.draw_networkx_nodes(
        g,
        pos,
        node_color=node_colors,
        node_size=520,
        edgecolors=node_edges,
        linewidths=[1.4 if n in orphans else 0.8 for n in g.nodes()],
        ax=ax,
    )
    nx.draw_networkx_labels(
        g,
        pos,
        labels={n: f"S{n}" for n in g.nodes()},
        font_size=8,
        font_color="white",
        ax=ax,
    )

    solid, virtual, barrier = _edge_buckets(dag)
    if solid:
        nx.draw_networkx_edges(
            g,
            pos,
            edgelist=solid,
            edge_color=PALETTE["barrier"],
            width=0.8,
            style="solid",
            arrows=True,
            arrowsize=10,
            ax=ax,
        )
    if virtual:
        nx.draw_networkx_edges(
            g,
            pos,
            edgelist=virtual,
            edge_color=PALETTE["virtual"],
            width=1.1,
            style="dashed",
            arrows=True,
            arrowsize=10,
            ax=ax,
        )
    if barrier:
        nx.draw_networkx_edges(
            g,
            pos,
            edgelist=barrier,
            edge_color=PALETTE["barrier"],
            width=1.0,
            style="dotted",
            arrows=True,
            arrowsize=10,
            ax=ax,
        )

    ax.set_axis_off()
    n_nodes = g.number_of_nodes()
    n_edges = g.number_of_edges()
    topo = f"{case.topo:.2f}" if case.topo is not None else "--"
    cont = f"{case.continuity:.2f}" if case.continuity is not None else "--"
    ax.set_title(
        f"{case.case_id}  ({case.source})\n"
        f"nodes={n_nodes}  edges={n_edges}  orphans={len(orphans)}  "
        f"q_topo={topo}  q_cont={cont}",
        fontsize=8,
        color=PALETTE["text"],
        loc="left",
        pad=4,
    )


def render_pack(cases: Sequence[Case], out_path: Path) -> None:
    n = len(cases)
    cols = 2 if n > 1 else 1
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(6.5, 2.4 * rows),
        squeeze=False,
    )
    for i, case in enumerate(cases):
        ax = axes[i // cols][i % cols]
        render_case_axes(ax, case)
    for j in range(n, rows * cols):
        axes[j // cols][j % cols].set_axis_off()
    legend_patches = [
        mpatches.Patch(color=PALETTE["node"], label="standard step"),
        mpatches.Patch(color=PALETTE["node_orphan"], label="orphan step"),
    ]
    fig.legend(
        handles=legend_patches,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=2,
        frameon=False,
        fontsize=7,
    )
    fig.suptitle(
        "TopoPRM reasoning DAGs — reference DAGs extracted from real training data",
        fontsize=9,
        color=PALETTE["text"],
        y=1.02,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def render_individual(cases: Sequence[Case], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for case in cases:
        fig, ax = plt.subplots(figsize=(4.0, 3.0))
        render_case_axes(ax, case)
        png = out_dir / f"case_{case.case_id}.png"
        fig.savefig(png, dpi=240, bbox_inches="tight", pad_inches=0.1)
        plt.close(fig)


def write_summary(cases: Sequence[Case], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# DAG case summary", ""]
    for case in cases:
        lines.append(f"## {case.case_id} — {case.source}")
        lines.append("")
        lines.append(f"- **Question**: {textwrap.shorten(case.question, 220, placeholder=' ...')}")
        lines.append(f"- **Gold answer**: {case.answer}")
        lines.append(f"- **nodes / edges / orphans**: {case.dag.graph.number_of_nodes()} / {case.dag.graph.number_of_edges()} / {len(case.dag.orphan_nodes())}")
        lines.append(f"- **q_topo / q_cont**: {case.topo:.3f} / {case.continuity:.3f}")
        lines.append("")
        lines.append("Steps:")
        for i, text in enumerate(case.raw_steps):
            lines.append(f"  - S{i}. {textwrap.shorten(text, 200, placeholder=' ...')}")
        lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=8, help="number of cases (5-10 recommended)")
    parser.add_argument("--source", default="mixed", help="gsm8k / math / mixed")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--jsonl", default=str(DEFAULT_SOURCE_JSONL))
    parser.add_argument(
        "--from-rollout",
        default=None,
        help="load from a rollout JSONL containing y_init / response text instead of reference DAGs",
    )
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--min-nodes", type=int, default=4)
    parser.add_argument("--max-nodes", type=int, default=9)
    args = parser.parse_args()

    if args.from_rollout:
        cases = load_cases_from_rollout(Path(args.from_rollout), n=args.n, seed=args.seed)
    else:
        cases = load_cases_from_training(
            Path(args.jsonl),
            n=args.n,
            source=args.source,
            min_nodes=args.min_nodes,
            max_nodes=args.max_nodes,
            seed=args.seed,
        )
    if not cases:
        raise SystemExit(
            "no cases loaded; try relaxing --min-nodes / --max-nodes or switch source"
        )

    out_dir = Path(args.out_dir)
    render_individual(cases, out_dir)
    render_pack(cases, out_dir / "pack.pdf")
    render_pack(cases, out_dir / "pack.png")
    write_summary(cases, out_dir / "summary.md")

    print(f"Rendered {len(cases)} case(s) to {out_dir}")
    for c in cases:
        print(
            f"  {c.case_id:<20} src={c.source:<6} n={c.dag.graph.number_of_nodes()} "

            f"e={c.dag.graph.number_of_edges()} orph={len(c.dag.orphan_nodes())} "
            f"q_topo={c.topo:.2f} q_cont={c.continuity:.2f}"
        )


if __name__ == "__main__":
    main()
