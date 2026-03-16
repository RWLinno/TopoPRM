import json

import pytest

from src.dag.node import Node, Edge, StepType, LocalVerdict
from src.dag.graph import ReasoningDAG
from src.dag.compress import merge_sequential_same_type, remove_transitive_edges, compress_dag


# ---------------------------------------------------------------------------
# Node / Edge dataclass roundtrip
# ---------------------------------------------------------------------------

class TestNodeEdgeDataclass:
    def test_node_to_dict_roundtrip(self):
        node = Node(
            step_id=0,
            raw_text="∵ x = 1",
            normalized_text="x = 1",
            exprs=["x = 1"],
            claims=[],
            step_type=StepType.DEFINITION,
            local_verdict=LocalVerdict.CORRECT,
            sub_question_id=1,
        )
        d = node.to_dict()
        restored = Node.from_dict(d)

        assert restored.step_id == node.step_id
        assert restored.raw_text == node.raw_text
        assert restored.normalized_text == node.normalized_text
        assert restored.exprs == node.exprs
        assert restored.claims == node.claims
        assert restored.step_type == node.step_type
        assert restored.local_verdict == node.local_verdict
        assert restored.sub_question_id == node.sub_question_id

    def test_edge_to_dict_roundtrip(self):
        edge = Edge(source=0, target=1, edge_type="dependency", dep_type="expr_ref", weight=1.0)
        d = edge.to_dict()
        restored = Edge.from_dict(d)

        assert restored.source == edge.source
        assert restored.target == edge.target
        assert restored.edge_type == edge.edge_type
        assert restored.dep_type == edge.dep_type
        assert restored.weight == edge.weight

    def test_node_defaults(self):
        node = Node(step_id=0, raw_text="text")
        assert node.normalized_text == ""
        assert node.exprs == []
        assert node.step_type == StepType.DERIVATION
        assert node.local_verdict == LocalVerdict.UNVERIFIABLE
        assert node.sub_question_id is None

    def test_edge_defaults(self):
        edge = Edge(source=0, target=1)
        assert edge.edge_type == "sequential"
        assert edge.dep_type == ""
        assert edge.weight == 0.5


# ---------------------------------------------------------------------------
# ReasoningDAG basics
# ---------------------------------------------------------------------------

def _make_three_node_dag() -> ReasoningDAG:
    dag = ReasoningDAG("test")
    for i in range(3):
        dag.add_node(Node(step_id=i, raw_text=f"step {i}"))
    dag.add_sequential_edges()
    return dag


class TestReasoningDAG:
    def test_empty_dag(self):
        dag = ReasoningDAG("empty")
        assert dag.num_nodes == 0
        assert dag.num_edges == 0
        assert dag.is_valid_dag()

    def test_add_nodes_and_edges(self):
        dag = _make_three_node_dag()
        assert dag.num_nodes == 3
        assert dag.num_edges == 2

    def test_add_dependency_edge(self):
        dag = _make_three_node_dag()
        dag.add_dependency_edge(0, 2, dep_type="expr_ref")
        assert dag.num_edges == 3

    def test_nodes_property(self):
        dag = _make_three_node_dag()
        nodes = dag.nodes
        assert len(nodes) == 3
        assert 0 in nodes and 1 in nodes and 2 in nodes

    def test_edges_property(self):
        dag = _make_three_node_dag()
        edges = dag.edges
        assert len(edges) == 2
        assert all(isinstance(e, Edge) for e in edges)

    def test_root_and_leaf_nodes(self):
        dag = _make_three_node_dag()
        assert dag.root_nodes() == [0]
        assert dag.leaf_nodes() == [2]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class TestValidateDAG:
    def test_validate_fields(self):
        dag = _make_three_node_dag()
        result = dag.validate_dag()

        expected_keys = {
            "is_acyclic", "is_connected", "isolated_nodes",
            "max_depth", "num_nodes", "num_edges", "has_orphan_conclusions",
        }
        assert set(result.keys()) == expected_keys

    def test_valid_dag_result(self):
        dag = _make_three_node_dag()
        result = dag.validate_dag()
        assert result["is_acyclic"] is True
        assert result["is_connected"] is True
        assert result["isolated_nodes"] == []
        assert result["num_nodes"] == 3
        assert result["num_edges"] == 2

    def test_isolated_node_detection(self):
        dag = ReasoningDAG("iso")
        for i in range(3):
            dag.add_node(Node(step_id=i, raw_text=f"step {i}"))
        dag.add_dependency_edge(0, 1)
        result = dag.validate_dag()
        assert 2 in result["isolated_nodes"]


# ---------------------------------------------------------------------------
# Cycle detection
# ---------------------------------------------------------------------------

class TestCycleDetection:
    def test_acyclic(self):
        dag = _make_three_node_dag()
        assert dag.has_cycles() is False
        assert dag.is_valid_dag() is True

    def test_cyclic(self):
        dag = _make_three_node_dag()
        dag.graph.add_edge(2, 0, weight=1.0, edge_type="dependency", dep_type="test")
        assert dag.has_cycles() is True
        assert dag.is_valid_dag() is False


# ---------------------------------------------------------------------------
# Topological order
# ---------------------------------------------------------------------------

class TestTopologicalOrder:
    def test_linear_chain(self):
        dag = _make_three_node_dag()
        order = dag.get_topological_order()
        assert order == [0, 1, 2]

    def test_preserves_dependency_ordering(self):
        dag = _make_three_node_dag()
        dag.add_dependency_edge(0, 2, dep_type="expr_ref")
        order = dag.get_topological_order()
        assert order.index(0) < order.index(2)

    def test_cyclic_raises(self):
        dag = _make_three_node_dag()
        dag.graph.add_edge(2, 0)
        with pytest.raises(Exception):
            dag.get_topological_order()


# ---------------------------------------------------------------------------
# JSON serialisation roundtrip
# ---------------------------------------------------------------------------

class TestSerialisation:
    def test_json_roundtrip(self):
        dag = ReasoningDAG("roundtrip")
        dag.add_node(Node(step_id=0, raw_text="∵ a=1", step_type=StepType.DEFINITION))
        dag.add_node(Node(step_id=1, raw_text="∴ b=2", step_type=StepType.DERIVATION))
        dag.add_sequential_edges()
        dag.add_dependency_edge(0, 1, dep_type="expr_ref")

        json_str = dag.to_json()
        restored = ReasoningDAG.from_json(json_str)

        assert restored.problem_id == dag.problem_id
        assert restored.num_nodes == dag.num_nodes
        assert restored.num_edges == dag.num_edges
        assert restored.nodes[0].step_type == StepType.DEFINITION
        assert restored.nodes[1].step_type == StepType.DERIVATION

    def test_to_dict_structure(self):
        dag = _make_three_node_dag()
        d = dag.to_dict()
        assert "problem_id" in d
        assert "nodes" in d
        assert "edges" in d
        assert len(d["nodes"]) == 3
        assert len(d["edges"]) == 2

    def test_from_dict_roundtrip(self):
        dag = _make_three_node_dag()
        d = dag.to_dict()
        restored = ReasoningDAG.from_dict(d)
        assert restored.num_nodes == dag.num_nodes
        assert restored.num_edges == dag.num_edges


# ---------------------------------------------------------------------------
# Compression
# ---------------------------------------------------------------------------

class TestCompression:
    def test_merge_sequential_same_type(self):
        dag = ReasoningDAG("merge")
        for i in range(3):
            dag.add_node(Node(
                step_id=i, raw_text=f"step {i}", step_type=StepType.DERIVATION,
            ))
        dag.add_sequential_edges()

        merged = merge_sequential_same_type(dag)
        assert merged.num_nodes == 1
        merged_node = list(merged.nodes.values())[0]
        assert "step 0" in merged_node.raw_text
        assert "step 2" in merged_node.raw_text

    def test_no_merge_different_types(self):
        dag = ReasoningDAG("no_merge")
        dag.add_node(Node(step_id=0, raw_text="s0", step_type=StepType.DEFINITION))
        dag.add_node(Node(step_id=1, raw_text="s1", step_type=StepType.DERIVATION))
        dag.add_node(Node(step_id=2, raw_text="s2", step_type=StepType.CONCLUSION))
        dag.add_sequential_edges()

        merged = merge_sequential_same_type(dag)
        assert merged.num_nodes == 3

    def test_remove_transitive_edges(self):
        dag = ReasoningDAG("transitive")
        for i in range(3):
            dag.add_node(Node(step_id=i, raw_text=f"step {i}"))
        dag.add_sequential_edges()
        dag.graph.add_edge(0, 2, weight=1.0, edge_type="dependency", dep_type="test")
        assert dag.num_edges == 3

        reduced = remove_transitive_edges(dag)
        assert reduced.num_edges == 2

    def test_compress_dag(self):
        dag = ReasoningDAG("compress")
        dag.add_node(Node(step_id=0, raw_text="s0", step_type=StepType.DERIVATION))
        dag.add_node(Node(step_id=1, raw_text="s1", step_type=StepType.DERIVATION))
        dag.add_node(Node(step_id=2, raw_text="s2", step_type=StepType.CONCLUSION))
        dag.add_sequential_edges()

        compressed = compress_dag(dag)
        assert compressed.num_nodes == 2
        assert compressed.num_edges == 1

    def test_compress_empty_dag(self):
        dag = ReasoningDAG("empty")
        compressed = compress_dag(dag)
        assert compressed.num_nodes == 0
        assert compressed.num_edges == 0
