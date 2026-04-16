import pytest

from src.data.build_dag import (
    canonicalize_expression,
    extract_steps_from_answer,
    extract_expressions,
    extract_claims,
    extract_variables,
    classify_step_type,
    build_dag_from_answer,
    parse_answer_to_dag_debug,
)
from src.dag.graph import ReasoningDAG
from src.dag.node import LocalVerdict, Node
from src.dag.node import StepType


# ---------------------------------------------------------------------------
# extract_steps_from_answer
# ---------------------------------------------------------------------------

class TestExtractSteps:
    def test_chinese_math_answer(self):
        answer = "∵ x > 0\n∴ x² > 0\n故 x² 是正数"
        steps = extract_steps_from_answer(answer)

        assert len(steps) == 3
        assert steps[0]["step_id"] == 0
        assert steps[0]["raw_text"] == "∵ x > 0"
        assert steps[1]["raw_text"] == "∴ x² > 0"
        assert steps[2]["raw_text"] == "故 x² 是正数"

    def test_empty_answer(self):
        assert extract_steps_from_answer("") == []

    def test_blank_lines_skipped(self):
        answer = "第一步\n\n\n第二步"
        steps = extract_steps_from_answer(answer)
        assert len(steps) == 2

    def test_sub_question_detection(self):
        answer = "【小题1】解：x=1\n【小题2】解：y=2"
        steps = extract_steps_from_answer(answer)
        assert len(steps) == 2
        assert steps[0]["sub_question_id"] == 1
        assert steps[1]["sub_question_id"] == 2

    def test_inline_step_markers(self):
        answer = "Step 1: let x=1; Step 2: therefore y=x+1; Step 3: answer y=2"
        steps = extract_steps_from_answer(answer)
        assert len(steps) >= 3


# ---------------------------------------------------------------------------
# extract_expressions
# ---------------------------------------------------------------------------

class TestExtractExpressions:
    def test_inline_dollar_math(self):
        exprs = extract_expressions("已知 $x=5$，求 $y=x+1$")
        assert "x=5" in exprs
        assert "y=x+1" in exprs

    def test_backslash_paren_math(self):
        exprs = extract_expressions(r"已知 \(a+b=3\) 则")
        assert "a+b=3" in exprs

    def test_equation_pattern(self):
        exprs = extract_expressions("因此 a = b + 1")
        assert any("a=b" in e for e in exprs)

    def test_var_assignment(self):
        exprs = extract_expressions("设 k＝2")
        assert any("k" in e for e in exprs)

    def test_canonicalize_expression(self):
        assert canonicalize_expression(" A ＝ B ") == "a=b"

    def test_no_expressions(self):
        assert extract_expressions("这是一段普通文字没有数学") == []


# ---------------------------------------------------------------------------
# extract_claims
# ---------------------------------------------------------------------------

class TestExtractClaims:
    def test_equality_pattern(self):
        claims = extract_claims("已知 x=y 是成立的")
        assert len(claims) >= 1

    def test_geometric_parallel(self):
        claims = extract_claims("AB∥CD")
        assert any("ab∥cd" in c for c in claims)

    def test_angle_equality(self):
        claims = extract_claims("∠ABC = 90°")
        assert any("∠abc" in c for c in claims)

    def test_because_therefore(self):
        claims = extract_claims("∵ x>0，∴ x²>0")
        assert len(claims) >= 1

    def test_incomplete_fragment_filtered(self):
        claims = extract_claims("所有有两种运输方案：")
        assert claims == []


class TestExtractVariables:
    def test_extract_basic_vars(self):
        vars_found = extract_variables("Let x=1, then y=x+2")
        assert "x" in vars_found and "y" in vars_found


# ---------------------------------------------------------------------------
# classify_step_type
# ---------------------------------------------------------------------------

class TestClassifyStepType:
    def test_because_is_definition(self):
        assert classify_step_type("∵ x > 0") == StepType.DEFINITION

    def test_known_is_definition(self):
        assert classify_step_type("已知 a = 3") == StepType.DEFINITION

    def test_therefore_is_derivation(self):
        assert classify_step_type("∴ y = x + 1") == StepType.DERIVATION

    def test_so_is_derivation(self):
        assert classify_step_type("所以 y = 2") == StepType.DERIVATION

    def test_solve_is_computation(self):
        assert classify_step_type("解得 x = 3") == StepType.COMPUTATION

    def test_simplify_is_computation(self):
        assert classify_step_type("化简得 2a") == StepType.COMPUTATION

    def test_gu_is_conclusion(self):
        assert classify_step_type("故 答案为5") == StepType.CONCLUSION

    def test_answer_is_conclusion(self):
        assert classify_step_type("答 x = 3") == StepType.CONCLUSION

    def test_substitute_is_substitution(self):
        assert classify_step_type("代入得 y = 4") == StepType.SUBSTITUTION

    def test_let_is_substitution(self):
        assert classify_step_type("令 t = x+1") == StepType.SUBSTITUTION

    def test_case_analysis(self):
        assert classify_step_type("分类讨论如下") == StepType.CASE_ANALYSIS

    def test_plain_equation_is_computation(self):
        assert classify_step_type("x = 3") == StepType.COMPUTATION

    def test_unknown_fallback(self):
        assert classify_step_type("这是普通文字") == StepType.UNKNOWN


# ---------------------------------------------------------------------------
# build_dag_from_answer
# ---------------------------------------------------------------------------

class TestBuildDagFromAnswer:
    def test_three_step_chain(self):
        answer = "∵ x = 3\n∴ y = x + 1\n故 y = 4"
        dag = build_dag_from_answer(answer, problem_id="test_001")

        assert dag.problem_id == "test_001"
        assert dag.num_nodes == 3
        assert dag.num_edges >= 2
        assert dag.is_valid_dag()

    def test_empty_answer(self):
        dag = build_dag_from_answer("", problem_id="empty")
        assert dag.num_nodes == 0
        assert dag.num_edges == 0

    def test_node_step_types(self):
        answer = "∵ a = 1\n计算 b = a + 1\n故 b = 2"
        dag = build_dag_from_answer(answer, problem_id="types")
        nodes = dag.nodes
        assert nodes[0].step_type == StepType.DEFINITION
        assert nodes[1].step_type == StepType.COMPUTATION
        assert nodes[2].step_type == StepType.CONCLUSION

    def test_sequential_edges_present(self):
        answer = "第一步\n第二步\n第三步"
        dag = build_dag_from_answer(answer)
        assert dag.num_nodes == 3
        edge_pairs = {(e.source, e.target) for e in dag.edges}
        assert (0, 1) in edge_pairs
        assert (1, 2) in edge_pairs

    def test_debug_payload_contains_evidence(self):
        answer = "设 x=1\n因此 y=x+1\n故 y=2"
        _, debug = parse_answer_to_dag_debug(answer)
        assert "steps" in debug and debug["steps"]
        assert "edges" in debug

    def test_hybrid_verdict_prefers_reference(self):
        answer = "设 x=1\n因此 y=x+1\n故 y=2"
        ref = ReasoningDAG("ref")
        ref.add_node(Node(step_id=0, raw_text="设 x=1", local_verdict=LocalVerdict.CORRECT))
        ref.add_node(Node(step_id=1, raw_text="因此 y=x+1", local_verdict=LocalVerdict.INCORRECT))
        ref.add_node(Node(step_id=2, raw_text="故 y=2", local_verdict=LocalVerdict.CORRECT))
        dag, _ = parse_answer_to_dag_debug(answer, reference_dag=ref.to_dict())
        assert dag.nodes[1].local_verdict == LocalVerdict.INCORRECT

    def test_non_chain_dependency_edge_exists(self):
        answer = "设 x=1\n由 x=1 得 y=2\n由 x=1 得 z=3\n故 y+z=5"
        dag, debug = parse_answer_to_dag_debug(answer)
        assert dag.num_nodes >= 4
        # Expect at least one dependency edge that is not simple order.
        dep_types = [d.get("dep_type", "") for _, _, d in dag.graph.edges(data=True)]
        assert any(t in {"expr_ref", "claim_ref", "var_ref", "expr_overlap"} for t in dep_types)
        assert "edge_source_stats" in debug["summary"]

    def test_sub_question_blocks_become_separate_components(self):
        answer = (
            "【小题1】设 x=1，列方程 x+1=2\n"
            "由 x+1=2 得 x=1\n"
            "【小题2】设 y=3，列方程 y-1=2\n"
            "由 y-1=2 得 y=3"
        )
        dag, debug = parse_answer_to_dag_debug(answer)
        assert debug["summary"]["sub_questions"] == [1, 2]
        # No edge should cross sub-question boundary.
        node_subq = {s["step_id"]: s["sub_question_id"] for s in debug["steps"]}
        for u, v, _ in dag.graph.edges(data=True):
            assert node_subq[u] == node_subq[v]
