from __future__ import annotations

import json
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

from src.dag.node import Edge, LocalVerdict, Node, StepType

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
                a, b, weight=0.5, edge_type="sequential", dep_type=""
            )

    def add_dependency_edge(
        self, src_id: int, tgt_id: int, dep_type: str = "logical"
    ) -> None:
        self.graph.add_edge(
            src_id, tgt_id, weight=1.0, edge_type="dependency", dep_type=dep_type
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
                    edge_type=data.get("edge_type", "sequential"),
                    dep_type=data.get("dep_type", ""),
                    weight=data.get("weight", 0.5),
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
                if d.get("edge_type") == "dependency"
            ]
        )
        if dep.number_of_edges() == 0:
            return 0
        return int(nx.dag_longest_path_length(dep))

    def orphan_nodes(self) -> list[int]:
        """Nodes with no dependency edges (only sequential or none)."""
        dep_participants: set[int] = set()
        for u, v, d in self.graph.edges(data=True):
            if d.get("edge_type") == "dependency":
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
            if d.get("edge_type") == "dependency"
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
                self.graph.edges[u, v].get("edge_type") != "dependency"
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
                edge_type=e.edge_type,
                dep_type=e.dep_type,
            )
        return dag

    @classmethod
    def from_json(cls, json_str: str) -> ReasoningDAG:
        return cls.from_dict(json.loads(json_str))

    # ---- visualisation ------------------------------------------------------

    def visualize(self, output_path: Optional[str] = None) -> None:
        fig, ax = plt.subplots(figsize=(12, 8))
        pos = nx.spring_layout(self.graph, seed=42)
        colors = [
            _STEP_COLOR.get(self._nodes[n].step_type, "#d9d9d9")
            for n in self.graph.nodes()
        ]

        nx.draw_networkx_nodes(self.graph, pos, node_color=colors, node_size=600, ax=ax)
        nx.draw_networkx_labels(self.graph, pos, font_size=8, ax=ax)

        dep_edges = [
            (u, v)
            for u, v, d in self.graph.edges(data=True)
            if d.get("edge_type") == "dependency"
        ]
        seq_edges = [
            (u, v)
            for u, v, d in self.graph.edges(data=True)
            if d.get("edge_type") != "dependency"
        ]
        nx.draw_networkx_edges(
            self.graph, pos, edgelist=seq_edges, style="dashed",
            alpha=0.4, edge_color="gray", ax=ax,
        )
        nx.draw_networkx_edges(
            self.graph, pos, edgelist=dep_edges, style="solid",
            edge_color="black", width=1.5, ax=ax,
        )

        ax.set_title(f"ReasoningDAG – {self.problem_id}")
        ax.axis("off")
        if output_path:
            fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
