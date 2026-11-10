from __future__ import annotations

import json
import math
import random
import re
from io import BytesIO
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import networkx as nx
import plotly.graph_objects as go
import streamlit as st

from src.data.build_dag import parse_answer_to_dag_debug
from src.dag.compress import compress_dag_by_layers
from src.dag.graph import ReasoningDAG
from src.reward.composite_reward import LengthReward, TopoCompositeReward
from src.reward.continuity_reward import ContinuityReward
from src.reward.format_reward import FormatReward
from src.reward.outcome_reward import OutcomeReward
from src.reward.topo_reward import TopoReward


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _extract_answer_block(solution: str) -> str:
    m = re.search(r"<answer>\s*(.*?)\s*</answer>", solution, re.DOTALL)
    return m.group(1).strip() if m else "{}"


def _safe_reference_dag(record: dict[str, Any]) -> dict[str, Any]:
    ref = record.get("reference_dag")
    if isinstance(ref, str):
        try:
            ref = json.loads(ref)
        except json.JSONDecodeError:
            ref = {}
    return ref or {}


def _debug_from_dag(dag: ReasoningDAG) -> dict[str, Any]:
    steps: list[dict[str, Any]] = []
    for sid in sorted(dag.nodes):
        n = dag.nodes[sid]
        steps.append(
            {
                "step_id": n.step_id,
                "raw_text": n.raw_text,
                "normalized_text": n.normalized_text,
                "exprs": n.exprs,
                "claims": n.claims,
                "step_type": n.step_type.value,
                "local_verdict": n.local_verdict.value,
                "sub_question_id": n.sub_question_id,
            }
        )

    edges: list[dict[str, Any]] = []
    for u, v, d in dag.graph.edges(data=True):
        edges.append(
            {
                "source": u,
                "target": v,
                "edge_type": d.get("edge_type", ""),
                "dep_type": d.get("dep_type", ""),
                "weight": d.get("weight", 0.0),
            }
        )

    return {
        "steps": steps,
        "edges": edges,
        "summary": {
            "num_nodes": dag.num_nodes,
            "num_edges": dag.num_edges,
            "is_acyclic": dag.is_valid_dag(),
            "num_roots": len(dag.root_nodes()),
            "num_leaves": len(dag.leaf_nodes()),
        },
    }


def _detect_record_format(record: dict[str, Any]) -> str:
    """Return 'training' (reference_dag), 'cached' (cached_dag), or 'rollout' (response)."""
    if isinstance(record.get("cached_dag"), dict) and record["cached_dag"].get("nodes"):
        return "cached"
    if isinstance(record.get("response"), str) and record["response"].strip():
        return "rollout"
    return "training"


def _build_dag_from_cached(cached: dict[str, Any], problem_id: str) -> tuple[ReasoningDAG, dict[str, Any]]:
    """Materialize a ReasoningDAG from a cached_dag dict produced by preprocess_dag_cache."""
    from src.dag.node import LocalVerdict, Node, StepType

    type_map = {t.value: t for t in StepType}
    verdict_map = {v.value: v for v in LocalVerdict}

    dag = ReasoningDAG(problem_id=problem_id)
    for n in cached.get("nodes", []):
        dag.add_node(
            Node(
                step_id=int(n.get("step_id", 0)),
                raw_text=str(n.get("raw_text", "")),
                normalized_text=str(n.get("normalized_text", n.get("raw_text", ""))),
                exprs=list(n.get("exprs", []) or []),
                claims=list(n.get("claims", []) or []),
                step_type=type_map.get(str(n.get("step_type", "")), StepType.UNKNOWN),
                local_verdict=verdict_map.get(
                    str(n.get("local_verdict", "")), LocalVerdict.UNVERIFIABLE
                ),
                sub_question_id=n.get("sub_question_id"),
            )
        )

    for e in cached.get("edges", []):
        try:
            u = int(e.get("source"))
            v = int(e.get("target"))
        except (TypeError, ValueError):
            continue
        if u not in dag.nodes or v not in dag.nodes:
            continue
        dep_type = str(e.get("dep_type", ""))
        edge_type = str(e.get("edge_type", "")) or None
        if edge_type and edge_type == "double_barrier_edge":
            dag.add_implicit_barrier_edge(u, v)
        elif edge_type and edge_type == "solid_edge":
            dag.graph.add_edge(u, v, weight=float(e.get("weight", 0.3)),
                               edge_type="solid_edge", dep_type=dep_type or "order")
        else:
            dag.add_dependency_edge(u, v, dep_type or "expr_ref")

    debug = _debug_from_dag(dag)
    debug["edges"] = [
        {
            "source": e.get("source"),
            "target": e.get("target"),
            "edge_type": e.get("edge_type", ""),
            "dep_type": e.get("dep_type", ""),
            "source_kind": e.get("source_kind", ""),
            "evidence": e.get("evidence", ""),
        }
        for e in cached.get("edges", [])
    ]
    return dag, debug


def _record_to_view_payload(
    record: dict[str, Any],
    sample_idx: int,
    dag_source: str = "reference_dag",
) -> dict[str, Any]:
    record_format = _detect_record_format(record)
    pid_default = f"sample_{sample_idx}"

    if record_format == "cached":
        cached = record["cached_dag"]
        problem_id = str(record.get("problem_id", record.get("question", pid_default))[:40])
        dag, debug = _build_dag_from_cached(cached, problem_id=problem_id)
        reasoning_text = record.get("response", "")
        answer_block = str(record.get("gold", ""))
        completion_text = f"<think>\n{reasoning_text}\n</think>\n<answer>\n{answer_block}\n</answer>"
    elif record_format == "rollout":
        reasoning_text = record.get("response", "")
        problem_id = str(record.get("problem_id", record.get("question", pid_default))[:40])
        dag, debug = parse_answer_to_dag_debug(reasoning_text, problem_id=problem_id)
        answer_block = str(record.get("gold", ""))
        completion_text = f"<think>\n{reasoning_text}\n</think>\n<answer>\n{answer_block}\n</answer>"
    else:
        ref = _safe_reference_dag(record)
        nodes = ref.get("nodes", [])
        reasoning_text = "\n".join(n.get("raw_text", "") for n in nodes if n.get("raw_text"))
        answer_block = _extract_answer_block(record.get("solution", ""))
        completion_text = f"<think>\n{reasoning_text}\n</think>\n<answer>\n{answer_block}\n</answer>"
        problem_id = ref.get("problem_id", pid_default)
        if dag_source == "reparse_think":
            dag, debug = parse_answer_to_dag_debug(reasoning_text, problem_id=problem_id)
        else:
            if ref.get("nodes") and ref.get("edges"):
                dag = ReasoningDAG.from_dict(ref)
                debug = _debug_from_dag(dag)
            else:
                dag, debug = parse_answer_to_dag_debug(reasoning_text, problem_id=problem_id)

    reward_inputs = [[{"content": completion_text}]]
    solution = record.get("solution")
    reference_dag = record.get("reference_dag")

    try:
        outcome = OutcomeReward()(reward_inputs, solution=solution)[0]
    except Exception:
        outcome = float(bool(record.get("correct_pass1")))
    try:
        fmt = FormatReward()(reward_inputs)[0]
    except Exception:
        fmt = 0.0
    try:
        length = LengthReward()(reward_inputs)[0]
    except Exception:
        length = 0.0
    topo_model = TopoReward()
    try:
        topo = topo_model(reward_inputs, reference_dag=reference_dag)[0]
        topo_diag = topo_model.last_diagnostics[0] if topo_model.last_diagnostics else {}
    except Exception:
        topo, topo_diag = 0.0, {}
    try:
        continuity = ContinuityReward()(reward_inputs)[0]
    except Exception:
        continuity = 0.0
    try:
        total = TopoCompositeReward()(reward_inputs, solution=solution, reference_dag=reference_dag)[0]
    except Exception:
        total = 0.0

    return {
        "sample_idx": sample_idx,
        "problem_id": problem_id,
        "record_format": record_format,
        "record": record,
        "reasoning_text": reasoning_text,
        "completion_text": completion_text,
        "dag": dag,
        "debug": debug,
        "reward": {
            "outcome": outcome,
            "format": fmt,
            "length": length,
            "topology": topo,
            "topology_terms": topo_diag,
            "continuity": continuity,
            "total": total,
        },
    }


def _graph_figure(dag: ReasoningDAG):
    fig, ax = plt.subplots(figsize=(9, 6))
    g = dag.graph
    pos = nx.spring_layout(g, seed=42)

    orphan = set(dag.orphan_nodes())
    node_colors = ["#ffaaaa" if n in orphan else "#9ecbff" for n in g.nodes()]
    nx.draw_networkx_nodes(g, pos, node_color=node_colors, node_size=850, ax=ax)
    nx.draw_networkx_labels(g, pos, font_size=8, ax=ax)

    weak_edges = []
    strong_edges = []
    invalid_edges = []
    for u, v, d in g.edges(data=True):
        et = d.get("edge_type", "")
        if dag.is_virtual_edge(et) and u >= v:
            invalid_edges.append((u, v))
        elif dag.is_virtual_edge(et):
            strong_edges.append((u, v))
        else:
            weak_edges.append((u, v))

    nx.draw_networkx_edges(
        g,
        pos,
        edgelist=weak_edges,
        edge_color="#888",
        alpha=0.5,
        width=1.0,
        arrows=True,
        arrowstyle="-|>",
        arrowsize=14,
        ax=ax,
    )
    nx.draw_networkx_edges(
        g,
        pos,
        edgelist=strong_edges,
        edge_color="#111",
        style="dashed",
        width=1.8,
        arrows=True,
        arrowstyle="-|>",
        arrowsize=16,
        ax=ax,
    )
    nx.draw_networkx_edges(
        g,
        pos,
        edgelist=invalid_edges,
        edge_color="#ff4d4f",
        style="solid",
        width=2.5,
        arrows=True,
        arrowstyle="-|>",
        arrowsize=18,
        ax=ax,
    )
    ax.set_axis_off()
    ax.set_title("Reasoning DAG")
    return fig


def _trim_text(text: str, max_len: int = 140) -> str:
    t = (text or "").replace("\n", " ").strip()
    if len(t) <= max_len:
        return t
    return t[: max_len - 3] + "..."


def _node_hover_text(node: Any) -> str:
    exprs = ", ".join(node.exprs[:5]) if getattr(node, "exprs", None) else "-"
    claims = ", ".join(node.claims[:3]) if getattr(node, "claims", None) else "-"
    return (
        f"id: {node.step_id}<br>"
        f"type: {node.step_type.value}<br>"
        f"verdict: {node.local_verdict.value}<br>"
        f"exprs: {exprs}<br>"
        f"claims: {claims}<br>"
        f"text: {_trim_text(node.raw_text, 220)}"
    )


def _layered_layout(g: nx.DiGraph) -> dict[Any, tuple[float, float]]:
    if not g.nodes:
        return {}
    if nx.is_directed_acyclic_graph(g):
        layers = list(nx.topological_generations(g))
        pos: dict[Any, tuple[float, float]] = {}
        for depth, layer in enumerate(layers):
            count = len(layer)
            for i, n in enumerate(sorted(layer)):
                y = -depth
                x = i - (count - 1) / 2.0
                pos[n] = (x, y)
        return pos
    return nx.spring_layout(g, seed=42)


def _shorten_segment(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    trim_start: float = 0.10,
    trim_end: float = 0.16,
) -> tuple[float, float, float, float]:
    dx = x1 - x0
    dy = y1 - y0
    length = math.hypot(dx, dy)
    if length < 1e-9:
        return x0, y0, x1, y1
    ux = dx / length
    uy = dy / length
    sx = x0 + ux * trim_start
    sy = y0 + uy * trim_start
    ex = x1 - ux * trim_end
    ey = y1 - uy * trim_end
    return sx, sy, ex, ey


def _quadratic_points(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    curvature: float = 0.0,
    n_points: int = 24,
) -> tuple[list[float], list[float]]:
    mx = (x0 + x1) / 2.0
    my = (y0 + y1) / 2.0
    dx = x1 - x0
    dy = y1 - y0
    length = math.hypot(dx, dy) + 1e-9
    nxv = -dy / length
    nyv = dx / length
    cx = mx + nxv * curvature
    cy = my + nyv * curvature

    xs: list[float] = []
    ys: list[float] = []
    for i in range(n_points):
        t = i / (n_points - 1)
        x = (1 - t) * (1 - t) * x0 + 2 * (1 - t) * t * cx + t * t * x1
        y = (1 - t) * (1 - t) * y0 + 2 * (1 - t) * t * cy + t * t * y1
        xs.append(x)
        ys.append(y)
    return xs, ys


def _plotly_dag_figure(
    dag: ReasoningDAG,
    title: str,
    layer_to_nodes: dict[int, list[int]] | None = None,
    node_to_layer: dict[int, int] | None = None,
    layer_chain_map: dict[int, int] | None = None,
) -> go.Figure:
    g = dag.graph
    pos = _layered_layout(g)
    orphan = set(dag.orphan_nodes())
    layer_to_nodes = layer_to_nodes or {}
    node_to_layer = node_to_layer or {}
    layer_chain_map = layer_chain_map or {}

    edge_traces: list[go.Scatter] = []
    arrow_annotations: list[dict[str, Any]] = []

    # Draw layer boxes (single-view hierarchy)
    shapes: list[dict[str, Any]] = []
    for li, nodes in layer_to_nodes.items():
        valid_nodes = [n for n in nodes if n in pos]
        if not valid_nodes:
            continue
        xs = [pos[n][0] for n in valid_nodes]
        ys = [pos[n][1] for n in valid_nodes]
        min_x, max_x = min(xs) - 0.45, max(xs) + 0.45
        min_y, max_y = min(ys) - 0.36, max(ys) + 0.36
        shapes.append(
            dict(
                type="rect",
                xref="x",
                yref="y",
                x0=min_x,
                y0=min_y,
                x1=max_x,
                y1=max_y,
                line=dict(color="#7d7d7d", width=1, dash="dot"),
                fillcolor="rgba(180,180,180,0.04)",
                layer="below",
            )
        )

    for u, v, d in g.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        et = d.get("edge_type", "")
        dep = d.get("dep_type", "")

        sx, sy, ex, ey = _shorten_segment(x0, y0, x1, y1)
        if dag.is_virtual_edge(et):
            color = "#111111"
            dash = "dash"
            width = 2.2
            curvature = 0.12
        elif dag.is_barrier_edge(et):
            color = "#9e9ac8"
            dash = "dot"
            width = 1.8
            curvature = -0.16
        else:
            color = "#a8a8a8"
            dash = "solid"
            width = 1.0
            curvature = 0.0

        # Separate sequential edge from dependency edge by type/layer
        if dag.is_solid_edge(et) and node_to_layer.get(u) == node_to_layer.get(v):
            curvature = -0.08

        exs, eys = _quadratic_points(sx, sy, ex, ey, curvature=curvature)

        edge_traces.append(
            go.Scatter(
                x=[*exs, None],
                y=[*eys, None],
                mode="lines",
                line=dict(color=color, width=width, dash=dash),
                hoverinfo="text",
                text=[f"{u} -> {v}<br>edge_type={et}<br>dep_type={dep}", "", ""],
                showlegend=False,
            )
        )
        arrow_annotations.append(
            dict(
                x=ex,
                y=ey,
                ax=exs[-4],
                ay=eys[-4],
                xref="x",
                yref="y",
                axref="x",
                ayref="y",
                showarrow=True,
                arrowhead=3,
                arrowsize=1.2,
                arrowwidth=1.2,
                arrowcolor=color,
                opacity=0.9,
            )
        )

    node_x: list[float] = []
    node_y: list[float] = []
    node_text: list[str] = []
    node_label: list[str] = []
    node_color: list[str] = []
    for n in g.nodes():
        x, y = pos[n]
        node_x.append(x)
        node_y.append(y)
        nd = dag.nodes.get(n)
        if nd is not None:
            node_text.append(_node_hover_text(nd))
        else:
            node_text.append(f"id: {n}")
        node_label.append(str(n))
        node_color.append("#ffaaaa" if n in orphan else "#9ecbff")

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=node_label,
        textposition="middle center",
        hovertemplate="%{customdata}<extra></extra>",
        customdata=node_text,
        marker=dict(size=34, color=node_color, line=dict(color="#2a2a2a", width=1)),
        showlegend=False,
    )

    # Layer labels and chain mapping overlay.
    text_x: list[float] = []
    text_y: list[float] = []
    text_v: list[str] = []
    for li, nodes in layer_to_nodes.items():
        valid_nodes = [n for n in nodes if n in pos]
        if not valid_nodes:
            continue
        xs = [pos[n][0] for n in valid_nodes]
        ys = [pos[n][1] for n in valid_nodes]
        tx = max(xs) + 0.55
        ty = sum(ys) / len(ys)
        chain_id = layer_chain_map.get(li, li)
        text_x.append(tx)
        text_y.append(ty)
        text_v.append(f"L{li} -> C{chain_id}")
    layer_text_trace = go.Scatter(
        x=text_x,
        y=text_y,
        mode="text",
        text=text_v,
        textposition="middle left",
        hoverinfo="skip",
        showlegend=False,
    )

    fig = go.Figure(data=[*edge_traces, node_trace, layer_text_trace])
    fig.update_layout(
        title=title,
        hovermode="closest",
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        shapes=shapes,
        annotations=arrow_annotations,
        height=560,
    )
    return fig


def _shortest_root_leaf_path(dag: ReasoningDAG) -> list[int]:
    g = dag.graph
    if g.number_of_nodes() == 0:
        return []
    roots = dag.root_nodes() or sorted(g.nodes())
    leaves = dag.leaf_nodes() or sorted(g.nodes())
    candidates: list[list[int]] = []
    for r in roots:
        for l in leaves:
            if r == l:
                continue
            if nx.has_path(g, r, l):
                candidates.append(nx.shortest_path(g, source=r, target=l))
    if not candidates:
        return dag.get_topological_order()[:1]
    candidates.sort(key=lambda p: (len(p), p))
    return candidates[0]


def _path_to_dag(dag: ReasoningDAG, path: list[int], problem_suffix: str) -> ReasoningDAG:
    sub = ReasoningDAG(problem_id=f"{dag.problem_id}_{problem_suffix}")
    for nid in path:
        node = dag.nodes.get(nid)
        if node is not None:
            sub.add_node(node)
    for u, v in zip(path, path[1:]):
        edge_data = dag.graph.edges[u, v] if dag.graph.has_edge(u, v) else {}
        sub.graph.add_edge(
            u,
            v,
            weight=edge_data.get("weight", 1.0),
            edge_type=edge_data.get("edge_type", "virtual_edge"),
            dep_type=edge_data.get("dep_type", "shortest_chain"),
        )
    return sub


def _edge_annotations(dag: ReasoningDAG) -> Dict[str, Any]:
    weak = []
    invalid = []
    for u, v, d in dag.graph.edges(data=True):
        et = d.get("edge_type", "")
        if dag.is_virtual_edge(et) and u >= v:
            invalid.append((u, v, et, d.get("dep_type", "")))
        elif not dag.is_virtual_edge(et):
            weak.append((u, v, et, d.get("dep_type", "")))
    return {
        "orphans": dag.orphan_nodes(),
        "weak_edges": weak,
        "invalid_dependencies": invalid,
    }


def main():
    st.set_page_config(page_title="TopoPRM DAG Viewer", layout="wide")
    st.title("TopoPRM DAG + Reward Visualization")

    data_path = st.sidebar.text_input(
        "Data path",
        "output/dag_audit/dag_audit_dr1_7b_gsm8k_cached.jsonl",
        help="Supports training jsonl (with reference_dag), rollout traces.jsonl "
        "(with response), or audit cached.jsonl (with cached_dag).",
    )
    dag_source = st.sidebar.selectbox(
        "DAG source (training data only)",
        options=["reference_dag", "reparse_think"],
        index=0,
        help="Has no effect on rollout / cached data; those use their own stored DAG.",
    )
    if "viewer_seed" not in st.session_state:
        st.session_state.viewer_seed = 42

    rows = _load_jsonl(Path(data_path))
    if not rows:
        st.error("The data file is empty or does not exist.")
        return

    benchmarks = sorted({str(r.get("benchmark", "")) for r in rows if r.get("benchmark")})
    if benchmarks:
        bench_choice = st.sidebar.selectbox(
            "Benchmark filter", options=["All"] + benchmarks, index=0
        )
        if bench_choice != "All":
            rows = [r for r in rows if str(r.get("benchmark", "")) == bench_choice]
        st.sidebar.caption(f"Samples after filtering: {len(rows)}")

    if not rows:
        st.warning("No samples for the selected benchmark.")
        return

    max_idx = max(len(rows) - 1, 0)
    sample_idx = st.sidebar.number_input("Sample index", min_value=0, max_value=max_idx, value=0, step=1)
    if st.sidebar.button("Random sample"):
        sample_idx = random.randint(0, max_idx)
        st.session_state.viewer_seed += 1

    payload = _record_to_view_payload(
        rows[int(sample_idx)],
        int(sample_idx),
        dag_source=dag_source,
    )
    dag = payload["dag"]
    debug = payload["debug"]
    st.sidebar.caption(f"record format: **{payload['record_format']}**")

    compressed_layer_dag, layer_to_nodes, _node_to_layer = compress_dag_by_layers(dag)
    layer_chain_map = {li: li for li in layer_to_nodes.keys()}
    compression_stats = {
        "orig_nodes": dag.num_nodes,
        "orig_edges": dag.num_edges,
        "compressed_chain_nodes": compressed_layer_dag.num_nodes,
        "compressed_chain_edges": compressed_layer_dag.num_edges,
        "num_layers": len(layer_to_nodes),
        "node_compression_ratio": round(
            (compressed_layer_dag.num_nodes / max(1, dag.num_nodes)), 4
        ),
        "edge_compression_ratio": round(
            (compressed_layer_dag.num_edges / max(1, dag.num_edges)), 4
        ),
        "layer_to_nodes": layer_to_nodes,
    }

    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Raw reasoning text")
        st.text_area("reasoning", payload["reasoning_text"], height=300)
        st.subheader("Step segmentation and intermediate artifacts")
        st.json(debug.get("steps", []))
    with col2:
        st.subheader("Single-view DAG (layer boxes + sequential/dependency edges + compression map)")
        st.plotly_chart(
            _plotly_dag_figure(
                dag,
                "Reasoning DAG (Layer-Box Single View)",
                layer_to_nodes=layer_to_nodes,
                node_to_layer=_node_to_layer,
                layer_chain_map=layer_chain_map,
            ),
            use_container_width=True,
        )
        st.subheader("Structural annotations")
        st.json(_edge_annotations(dag))

    st.subheader("Node features (exprs / claims / step_type / verdict)")
    node_feature_rows = []
    for sid in sorted(dag.nodes):
        n = dag.nodes[sid]
        node_feature_rows.append(
            {
                "step_id": n.step_id,
                "step_type": getattr(n.step_type, "value", str(n.step_type)),
                "local_verdict": getattr(n.local_verdict, "value", str(n.local_verdict)),
                "exprs": list(n.exprs)[:6],
                "claims": list(n.claims)[:3],
                "raw_text": (n.raw_text or "")[:160],
            }
        )
    st.dataframe(node_feature_rows, use_container_width=True, hide_index=True)

    st.subheader("Dependency edges (dep_type / source_kind / evidence)")
    edge_rows = []
    cached_edges = {
        (int(e.get("source", -1)), int(e.get("target", -1))): e
        for e in (debug.get("edges", []) if isinstance(debug.get("edges"), list) else [])
        if isinstance(e, dict)
    }
    for u, v, d in dag.graph.edges(data=True):
        cached = cached_edges.get((u, v), {})
        edge_rows.append(
            {
                "source": u,
                "target": v,
                "edge_type": d.get("edge_type", ""),
                "dep_type": d.get("dep_type", "") or cached.get("dep_type", ""),
                "source_kind": cached.get("source_kind", ""),
                "weight": float(d.get("weight", 0.0)),
                "evidence": cached.get("evidence", "")[:200],
            }
        )
    if edge_rows:
        st.dataframe(edge_rows, use_container_width=True, hide_index=True)
    else:
        st.info("The current DAG has no edges.")

    st.subheader("Reward components (computed with the real reward logic)")
    st.json(payload["reward"])
    st.subheader("Layer-compression round-trip statistics")
    st.json(compression_stats)

    st.subheader("Graph JSON export")
    graph_json = json.dumps(dag.to_dict(), ensure_ascii=False, indent=2)
    st.download_button(
        label="Download graph.json",
        data=graph_json.encode("utf-8"),
        file_name=f"{payload['problem_id']}_graph.json",
        mime="application/json",
    )

    # PNG export from rendered figure
    png = BytesIO()
    fig = _graph_figure(dag)
    fig.savefig(png, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    st.download_button(
        label="Download DAG image",
        data=png.getvalue(),
        file_name=f"{payload['problem_id']}_dag.png",
        mime="image/png",
    )


if __name__ == "__main__":
    main()
