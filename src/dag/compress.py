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
            weight=original_data.get("weight", 0.3),
            edge_type=original_data.get("edge_type", "solid_edge"),
            dep_type=original_data.get("dep_type", ""),
        )

    return new_dag


def compress_dag(dag: ReasoningDAG) -> ReasoningDAG:
    """Apply merge_sequential_same_type then remove_transitive_edges."""
    dag = merge_sequential_same_type(dag)
    dag = remove_transitive_edges(dag)
    return dag


def compress_dag_by_layers(
    dag: ReasoningDAG,
) -> tuple[ReasoningDAG, dict[int, list[int]], dict[int, int]]:
    """Compress DAG by topological layers.

    Returns:
      - compressed chain DAG (layer i -> layer i+1)
      - layer_to_nodes mapping
      - node_to_layer mapping
    """
    compressed = ReasoningDAG(problem_id=f"{dag.problem_id}_layer_compress")
    if dag.num_nodes == 0:
        return compressed, {}, {}

    if dag.is_valid_dag():
        layers = list(nx.topological_generations(dag.graph))
        layer_to_nodes = {
            li: sorted(int(n) for n in layer)
            for li, layer in enumerate(layers)
        }
    else:
        order = sorted(dag.nodes.keys())
        layer_to_nodes = {i: [sid] for i, sid in enumerate(order)}

    node_to_layer: dict[int, int] = {}
    for li, node_ids in layer_to_nodes.items():
        for sid in node_ids:
            node_to_layer[sid] = li

    for li, node_ids in layer_to_nodes.items():
        texts = [dag.nodes[sid].raw_text for sid in node_ids if sid in dag.nodes]
        all_exprs: list[str] = []
        all_claims: list[str] = []
        for sid in node_ids:
            n = dag.nodes[sid]
            all_exprs.extend(n.exprs)
            all_claims.extend(n.claims)
        representative = dag.nodes[node_ids[0]]
        compressed.add_node(
            Node(
                step_id=li,
                raw_text="\n".join(texts),
                normalized_text=representative.normalized_text,
                exprs=all_exprs,
                claims=all_claims,
                step_type=representative.step_type,
                local_verdict=representative.local_verdict,
                sub_question_id=representative.sub_question_id,
            )
        )

    # Build chain over layers as shortest compressed reasoning backbone.
    layer_ids = sorted(layer_to_nodes.keys())
    for a, b in zip(layer_ids, layer_ids[1:]):
        compressed.graph.add_edge(
            a,
            b,
            weight=1.0,
            edge_type="virtual_edge",
            dep_type="layer_chain",
        )

    return compressed, layer_to_nodes, node_to_layer
