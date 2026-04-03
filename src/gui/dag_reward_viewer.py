from __future__ import annotations

import json
import random
import re
from io import BytesIO
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import networkx as nx
import streamlit as st

from src.data.build_dag import parse_answer_to_dag_debug
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


def _record_to_view_payload(record: dict[str, Any], sample_idx: int) -> dict[str, Any]:
    ref = record.get("reference_dag")
    if isinstance(ref, str):
        ref = json.loads(ref)
    ref = ref or {}

    nodes = ref.get("nodes", [])
    reasoning_text = "\n".join(n.get("raw_text", "") for n in nodes if n.get("raw_text"))
    answer_block = _extract_answer_block(record.get("solution", ""))
    completion_text = f"<think>\n{reasoning_text}\n</think>\n<answer>\n{answer_block}\n</answer>"

    dag, debug = parse_answer_to_dag_debug(reasoning_text, problem_id=ref.get("problem_id", f"sample_{sample_idx}"))
    reward_inputs = [[{"content": completion_text}]]
    solution = record.get("solution")
    reference_dag = record.get("reference_dag")

    outcome = OutcomeReward()(reward_inputs, solution=solution)[0]
    fmt = FormatReward()(reward_inputs)[0]
    length = LengthReward()(reward_inputs)[0]
    topo = TopoReward()(reward_inputs, reference_dag=reference_dag)[0]
    continuity = ContinuityReward()(reward_inputs)[0]
    total = TopoCompositeReward()(reward_inputs, solution=solution, reference_dag=reference_dag)[0]

    return {
        "sample_idx": sample_idx,
        "problem_id": ref.get("problem_id", f"sample_{sample_idx}"),
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

    nx.draw_networkx_edges(g, pos, edgelist=weak_edges, edge_color="#888", alpha=0.5, width=1.0, ax=ax)
    nx.draw_networkx_edges(g, pos, edgelist=strong_edges, edge_color="#111", style="dashed", width=1.8, ax=ax)
    nx.draw_networkx_edges(g, pos, edgelist=invalid_edges, edge_color="#ff4d4f", style="solid", width=2.5, ax=ax)
    ax.set_axis_off()
    ax.set_title("Reasoning DAG")
    return fig


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
    st.title("TopoPRM DAG + Reward 可视化")

    data_path = st.sidebar.text_input("数据路径", "data/grpo_ready/train.jsonl")
    if "viewer_seed" not in st.session_state:
        st.session_state.viewer_seed = 42

    rows = _load_jsonl(Path(data_path))
    max_idx = max(len(rows) - 1, 0)
    sample_idx = st.sidebar.number_input("样本索引", min_value=0, max_value=max_idx, value=0, step=1)
    if st.sidebar.button("随机刷新"):
        sample_idx = random.randint(0, max_idx)
        st.session_state.viewer_seed += 1

    payload = _record_to_view_payload(rows[int(sample_idx)], int(sample_idx))
    dag = payload["dag"]
    debug = payload["debug"]

    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("原始推理文本")
        st.text_area("reasoning", payload["reasoning_text"], height=300)
        st.subheader("Step 切分与中间产物")
        st.json(debug.get("steps", []))
    with col2:
        st.subheader("DAG 图")
        fig = _graph_figure(dag)
        st.pyplot(fig, clear_figure=True)
        st.subheader("结构标注")
        st.json(_edge_annotations(dag))

    st.subheader("Reward 分量（真实逻辑计算）")
    st.json(payload["reward"])

    st.subheader("Graph JSON 导出")
    graph_json = json.dumps(dag.to_dict(), ensure_ascii=False, indent=2)
    st.download_button(
        label="下载 graph.json",
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
        label="下载 DAG 图片",
        data=png.getvalue(),
        file_name=f"{payload['problem_id']}_dag.png",
        mime="image/png",
    )


if __name__ == "__main__":
    main()
