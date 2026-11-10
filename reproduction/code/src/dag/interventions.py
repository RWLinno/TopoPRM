"""Prespecified graph-information controls; the extractor and score stay fixed."""
from __future__ import annotations

import copy
import hashlib
import random


def intervene_graph(dag, mode: str, *, text: str, seed: int):
    if mode not in {"none", "local", "random_forward"}:
        raise ValueError(f"Unknown graph intervention: {mode}")
    if mode == "none":
        return dag
    result = copy.deepcopy(dag)
    nodes = sorted(result.nodes)
    edges = sorted(dag.graph.edges(data=True), key=lambda e: (e[0], e[1]))
    result.graph.remove_edges_from(list(result.graph.edges))
    if mode == "local":
        adjacent = set(zip(nodes, nodes[1:]))
        result.graph.add_edges_from((u, v, dict(data)) for u, v, data in edges if (u, v) in adjacent)
    else:
        possible = [(u, v) for i, u in enumerate(nodes) for v in nodes[i + 1:]]
        if any(u >= v for u, v, _ in edges) or len(edges) > len(possible):
            raise ValueError("Random-forward control requires the frozen forward-only extractor")
        digest = hashlib.sha256(f"{seed}:{text}".encode()).digest()
        rng = random.Random(int.from_bytes(digest[:8], "big"))
        targets = rng.sample(possible, len(edges))
        attributes = [dict(data) for _, _, data in edges]
        rng.shuffle(attributes)
        result.graph.add_edges_from((u, v, data) for (u, v), data in zip(targets, attributes))
        assert result.num_nodes == dag.num_nodes and result.num_edges == dag.num_edges
    assert result.is_valid_dag()
    return result
