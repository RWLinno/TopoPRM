from __future__ import annotations

import networkx as nx

from src.dag.graph import ReasoningDAG
from src.dag.node import Edge, Node


def merge_sequential_same_type(dag: ReasoningDAG) -> ReasoningDAG:
    """Merge chains of consecutive nodes that share the same StepType."""
    new_dag = ReasoningDAG(problem_id=dag.problem_id)
    nodes = dag.nodes
    ids = sorted(nodes)
    if not ids:
        return new_dag

    groups: list[list[int]] = [[ids[0]]]
    for sid in ids[1:]:
        prev_id = groups[-1][-1]
        if nodes[sid].step_type == nodes[prev_id].step_type:
            groups[-1].append(sid)
        else:
            groups.append([sid])

    id_map: dict[int, int] = {}
    for group in groups:
        representative = group[0]
        merged_text = "\n".join(nodes[sid].raw_text for sid in group)
        merged_exprs: list[str] = []
        merged_claims: list[str] = []
        for sid in group:
            merged_exprs.extend(nodes[sid].exprs)
            merged_claims.extend(nodes[sid].claims)

        new_node = Node(
            step_id=representative,
            raw_text=merged_text,
            normalized_text=nodes[representative].normalized_text,
            exprs=merged_exprs,
            claims=merged_claims,
            step_type=nodes[representative].step_type,
            local_verdict=nodes[representative].local_verdict,
            sub_question_id=nodes[representative].sub_question_id,
        )
        new_dag.add_node(new_node)
        for sid in group:
            id_map[sid] = representative

    for edge in dag.edges:
        src = id_map.get(edge.source, edge.source)
        tgt = id_map.get(edge.target, edge.target)
        if src == tgt:
            continue
        if not new_dag.graph.has_edge(src, tgt):
            new_dag.graph.add_edge(
                src, tgt,
                weight=edge.weight,
                edge_type=edge.edge_type,
                dep_type=edge.dep_type,
            )

    return new_dag


def remove_transitive_edges(dag: ReasoningDAG) -> ReasoningDAG:
    """Remove transitive edges via networkx transitive_reduction."""
    new_dag = ReasoningDAG(problem_id=dag.problem_id)
    for sid, node in dag.nodes.items():
        new_dag.add_node(node)

    if dag.num_edges == 0:
        return new_dag

    reduced = nx.transitive_reduction(dag.graph)

    for u, v in reduced.edges():
        original_data = dag.graph.edges[u, v]
        new_dag.graph.add_edge(
            u, v,
            weight=original_data.get("weight", 0.5),
            edge_type=original_data.get("edge_type", "sequential"),
            dep_type=original_data.get("dep_type", ""),
        )

    return new_dag


def compress_dag(dag: ReasoningDAG) -> ReasoningDAG:
    """Apply merge_sequential_same_type then remove_transitive_edges."""
    dag = merge_sequential_same_type(dag)
    dag = remove_transitive_edges(dag)
    return dag
