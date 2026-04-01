from __future__ import annotations

import json
from typing import Optional

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # matplotlib is optional for non-visual evaluation
    matplotlib = None
    plt = None
import networkx as nx

from src.dag.node import Edge, Node, StepType

_STEP_COLOR = {
    StepType.DEFINITION: "#6baed6",
    StepType.DERIVATION: "#74c476",
    StepType.COMPUTATION: "#fd8d3c",
    StepType.CONCLUSION: "#e6550d",
    StepType.AUXILIARY: "#bdbdbd",
    StepType.SUBSTITUTION: "#9e9ac8",
    StepType.CASE_ANALYSIS: "#f768a1",
    StepType.UNKNOWN: "#d9d9d9",
}

# Canonical edge semantics used by TopoPRM v2:
# - solid_edge: sequential dependency edge (weak signal)
# - virtual_edge: conditional dependency edge (strong signal)
# - double_barrier_edge: implicit blocking edge (no direct reward)
SOLID_EDGE = "solid_edge"
VIRTUAL_EDGE = "virtual_edge"
DOUBLE_BARRIER_EDGE = "double_barrier_edge"

# Backward-compatible aliases for legacy artifacts and old cached DAG files.
_SOLID_ALIASES = {"sequential", SOLID_EDGE}
_VIRTUAL_ALIASES = {"dependency", VIRTUAL_EDGE}
_BARRIER_ALIASES = {"implicit", DOUBLE_BARRIER_EDGE}


class ReasoningDAG:
    def __init__(self, problem_id: str) -> None:
        self.problem_id = problem_id
        self.graph = nx.DiGraph()
        self._nodes: dict[int, Node] = {}

    # ---- mutation -----------------------------------------------------------

    def add_node(self, node: Node) -> None:
        self._nodes[node.step_id] = node
        self.graph.add_node(node.step_id)

    def add_sequential_edges(self) -> None:
        ids = sorted(self._nodes)
        for a, b in zip(ids, ids[1:]):
            self.graph.add_edge(
                a, b, weight=0.3, edge_type=SOLID_EDGE, dep_type="order"
            )

    def add_dependency_edge(
        self, src_id: int, tgt_id: int, dep_type: str = "logical"
    ) -> None:
        self.graph.add_edge(
            src_id, tgt_id, weight=1.0, edge_type=VIRTUAL_EDGE, dep_type=dep_type
        )

    def add_implicit_barrier_edge(self, src_id: int, tgt_id: int) -> None:
        self.graph.add_edge(
            src_id,
            tgt_id,
            weight=0.0,
            edge_type=DOUBLE_BARRIER_EDGE,
            dep_type="implicit_block",
        )

    # ---- properties ---------------------------------------------------------

    @property
    def nodes(self) -> dict[int, Node]:
        return dict(self._nodes)

    @property
    def edges(self) -> list[Edge]:
        result: list[Edge] = []
        for u, v, data in self.graph.edges(data=True):
            result.append(
                Edge(
                    source=u,
                    target=v,
                    edge_type=data.get("edge_type", SOLID_EDGE),
                    dep_type=data.get("dep_type", ""),
                    weight=data.get("weight", 0.3),
                )
            )
        return result

    @property
    def num_nodes(self) -> int:
        return self.graph.number_of_nodes()

    @property
    def num_edges(self) -> int:
        return self.graph.number_of_edges()

    # ---- queries ------------------------------------------------------------

    @staticmethod
    def is_solid_edge(edge_type: str) -> bool:
        return edge_type in _SOLID_ALIASES

    @staticmethod
    def is_virtual_edge(edge_type: str) -> bool:
        return edge_type in _VIRTUAL_ALIASES

    @staticmethod
    def is_barrier_edge(edge_type: str) -> bool:
        return edge_type in _BARRIER_ALIASES

    def is_valid_dag(self) -> bool:
        return nx.is_directed_acyclic_graph(self.graph)

    def has_cycles(self) -> bool:
        return not nx.is_directed_acyclic_graph(self.graph)

    def get_topological_order(self) -> list[int]:
        return list(nx.topological_sort(self.graph))

    def get_dependency_depth(self) -> int:
        dep = self.graph.edge_subgraph(
            [
                (u, v)
                for u, v, d in self.graph.edges(data=True)
                if self.is_virtual_edge(d.get("edge_type", ""))
            ]
        )
        if dep.number_of_edges() == 0:
            return 0
        return int(nx.dag_longest_path_length(dep))

    def orphan_nodes(self) -> list[int]:
        """Nodes with no strong (virtual) dependency edges."""
        dep_participants: set[int] = set()
        for u, v, d in self.graph.edges(data=True):
            if self.is_virtual_edge(d.get("edge_type", "")):
                dep_participants.add(u)
                dep_participants.add(v)
        return sorted(n for n in self._nodes if n not in dep_participants)

    def root_nodes(self) -> list[int]:
        return sorted(n for n in self.graph.nodes() if self.graph.in_degree(n) == 0)

    def leaf_nodes(self) -> list[int]:
        return sorted(n for n in self.graph.nodes() if self.graph.out_degree(n) == 0)

    def direction_consistency(self) -> float:
        dep_edges = [
            (u, v)
            for u, v, d in self.graph.edges(data=True)
            if self.is_virtual_edge(d.get("edge_type", ""))
        ]
        if not dep_edges:
            return 1.0
        forward = sum(1 for u, v in dep_edges if u < v)
        return forward / len(dep_edges)

    # ---- validation ---------------------------------------------------------

    def validate_dag(self) -> dict:
        is_acyclic = self.is_valid_dag()
        undirected = self.graph.to_undirected()
        is_connected = nx.is_connected(undirected) if undirected.number_of_nodes() > 0 else True
        isolated = list(nx.isolates(self.graph))
        max_depth = (
            nx.dag_longest_path_length(self.graph) if is_acyclic and self.num_nodes > 0 else 0
        )
        conclusion_ids = [
            sid
            for sid, n in self._nodes.items()
            if n.step_type == StepType.CONCLUSION
        ]
        has_orphan_conclusions = any(
            all(
                not self.is_virtual_edge(self.graph.edges[u, v].get("edge_type", ""))
                for u, v in self.graph.in_edges(cid)
            )
            and self.graph.in_degree(cid) > 0
            for cid in conclusion_ids
        ) if conclusion_ids else False

        return {
            "is_acyclic": is_acyclic,
            "is_connected": is_connected,
            "isolated_nodes": isolated,
            "max_depth": max_depth,
            "num_nodes": self.num_nodes,
            "num_edges": self.num_edges,
            "has_orphan_conclusions": has_orphan_conclusions,
        }

    # ---- serialisation ------------------------------------------------------

    @staticmethod
    def _normalize_edge_type(edge_type: str) -> str:
        if edge_type in _SOLID_ALIASES:
            return SOLID_EDGE
        if edge_type in _VIRTUAL_ALIASES:
            return VIRTUAL_EDGE
        if edge_type in _BARRIER_ALIASES:
            return DOUBLE_BARRIER_EDGE
        return edge_type

    def to_dict(self) -> dict:
        return {
            "problem_id": self.problem_id,
            "nodes": [n.to_dict() for n in self._nodes.values()],
            "edges": [e.to_dict() for e in self.edges],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    @classmethod
    def from_dict(cls, data: dict) -> ReasoningDAG:
        dag = cls(problem_id=data["problem_id"])
        for nd in data["nodes"]:
            dag.add_node(Node.from_dict(nd))
        for ed in data["edges"]:
            e = Edge.from_dict(ed)
            dag.graph.add_edge(
                e.source,
                e.target,
                weight=e.weight,
                edge_type=cls._normalize_edge_type(e.edge_type),
                dep_type=e.dep_type,
            )
        return dag

    @classmethod
    def from_json(cls, json_str: str) -> ReasoningDAG:
        return cls.from_dict(json.loads(json_str))

    # ---- visualisation ------------------------------------------------------

    def visualize(self, output_path: Optional[str] = None) -> None:
        if plt is None:
            raise RuntimeError("matplotlib is required for DAG visualization")
        fig, ax = plt.subplots(figsize=(12, 8))
        pos = nx.spring_layout(self.graph, seed=42)
        colors = [
            _STEP_COLOR.get(self._nodes[n].step_type, "#d9d9d9")
            for n in self.graph.nodes()
        ]

        nx.draw_networkx_nodes(self.graph, pos, node_color=colors, node_size=600, ax=ax)
        nx.draw_networkx_labels(self.graph, pos, font_size=8, ax=ax)

        virtual_edges = [
            (u, v)
            for u, v, d in self.graph.edges(data=True)
            if self.is_virtual_edge(d.get("edge_type", ""))
        ]
        solid_edges = [
            (u, v)
            for u, v, d in self.graph.edges(data=True)
            if self.is_solid_edge(d.get("edge_type", ""))
        ]
        barrier_edges = [
            (u, v)
            for u, v, d in self.graph.edges(data=True)
            if self.is_barrier_edge(d.get("edge_type", ""))
        ]
        nx.draw_networkx_edges(
            self.graph,
            pos,
            edgelist=solid_edges,
            style="solid",
            alpha=0.35,
            edge_color="gray",
            width=1.0,
            ax=ax,
        )
        nx.draw_networkx_edges(
            self.graph,
            pos,
            edgelist=virtual_edges,
            style="dashed",
            edge_color="black",
            width=1.6,
            ax=ax,
        )
        nx.draw_networkx_edges(
            self.graph,
            pos,
            edgelist=barrier_edges,
            style="dashdot",
            edge_color="#9e9ac8",
            alpha=0.85,
            width=1.2,
            ax=ax,
        )

        ax.set_title(f"ReasoningDAG – {self.problem_id}")
        ax.axis("off")
        if output_path:
            fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
