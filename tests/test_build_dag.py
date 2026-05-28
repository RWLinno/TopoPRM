from pathlib import Path

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

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "dag_audit"


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
        node_subq = {s["step_id"]: s["sub_question_id"] for s in debug["steps"]}
        for u, v, _ in dag.graph.edges(data=True):
            assert node_subq[u] == node_subq[v]


# ---------------------------------------------------------------------------
# LLM-assisted hybrid dependency edges (offline preprocessing only)
# ---------------------------------------------------------------------------


class TestLLMHybridDependencyEdges:
    """Tests for the LLM-refined edges merged on top of rule edges.

    The runtime / GRPO path must never trigger the LLM; these tests use the
    explicit ``llm_client`` argument that the offline preprocessor exposes.
    """

    def _parsed_steps(self, answer: str):
        from src.data.build_dag import (
            ParsedStep,
            classify_step_type,
            extract_claim_keys,
            extract_claims,
            extract_expressions,
            extract_steps_from_answer,
            extract_variables,
        )
        import unicodedata

        parsed = []
        for s in extract_steps_from_answer(answer):
            text = s["raw_text"]
            norm = unicodedata.normalize("NFKC", text).strip()
            parsed.append(
                ParsedStep(
                    step_id=s["step_id"],
                    raw_text=text,
                    normalized_text=norm,
                    sub_question_id=s.get("sub_question_id"),
                    exprs=extract_expressions(text),
                    claims=extract_claims(text),
                    claim_keys=extract_claim_keys(text),
                    variables=extract_variables(norm),
                    step_type=classify_step_type(text),
                )
            )
        return parsed

    def test_default_runtime_path_skips_llm(self, monkeypatch):
        """Online path must remain rule-only when env flag is unset/False."""
        from src.data.build_dag import build_dependency_edges_by_llm

        monkeypatch.delenv("TOPO_DAG_LLM_REFINE", raising=False)
        steps = self._parsed_steps("Step 1: Let x=1.\nStep 2: Then y=x+1.")
        edges, _ = build_dependency_edges_by_llm(steps, llm_client=None)
        for src, tgt, _etype, dep in edges:
            assert dep != "llm_semantic" and dep != "llm_subgoal"

    def test_llm_edges_merged_with_confidence(self, monkeypatch):
        """A stub LLM client returning a strict-JSON edge should merge cleanly."""
        from src.data.build_dag import build_dependency_edges_by_llm

        monkeypatch.setenv("TOPO_DAG_LLM_REFINE", "1")
        steps = self._parsed_steps(
            "Step 1: Let x=1.\nStep 2: Define helper claim P.\nStep 3: Conclude y=x+1."
        )

        class StubClient:
            def generate(self, prompt: str) -> str:
                return (
                    "[{\"source\":1,\"target\":2,\"dep_type\":\"llm_semantic\","
                    "\"confidence\":0.7,\"evidence\":\"step 3 reuses helper claim P\"}]"
                )

        edges, evidences = build_dependency_edges_by_llm(steps, llm_client=StubClient())
        llm_edges = [
            (s, t, dep)
            for s, t, _et, dep in edges
            if dep in {"llm_semantic", "llm_subgoal"}
        ]
        assert (1, 2, "llm_semantic") in llm_edges
        llm_evs = [e for e in evidences if e.dep_type == "llm_semantic"]
        assert llm_evs and "conf=0.70" in llm_evs[0].evidence

    def test_llm_edges_rejected_when_violating_constraints(self, monkeypatch):
        """source>=target, unknown dep_type, and bad ids must be filtered."""
        from src.data.build_dag import build_dependency_edges_by_llm

        monkeypatch.setenv("TOPO_DAG_LLM_REFINE", "1")
        steps = self._parsed_steps(
            "Step 1: Let x=1.\nStep 2: Then y=x+1.\nStep 3: Conclude y=2."
        )

        class BadClient:
            def generate(self, prompt: str) -> str:
                return (
                    "["
                    "{\"source\":2,\"target\":1,\"dep_type\":\"llm_semantic\",\"confidence\":0.9,\"evidence\":\"backward\"},"
                    "{\"source\":0,\"target\":2,\"dep_type\":\"unknown_kind\",\"confidence\":0.4,\"evidence\":\"clamped\"},"
                    "{\"source\":99,\"target\":100,\"dep_type\":\"llm_semantic\",\"confidence\":0.5,\"evidence\":\"oob\"}"
                    "]"
                )

        edges, _ = build_dependency_edges_by_llm(steps, llm_client=BadClient())
        deps = [(s, t, dep) for s, t, _et, dep in edges if dep.startswith("llm_")]
        # Backward edge dropped, OOB edge dropped, unknown_kind clamped to llm_semantic.
        assert (2, 1, "llm_semantic") not in deps
        assert (99, 100, "llm_semantic") not in deps
        assert (0, 2, "llm_semantic") in deps

    def test_llm_failure_falls_back_to_rule_edges(self, monkeypatch, caplog):
        """If the LLM client raises, the pipeline must still return rule edges."""
        from src.data.build_dag import build_dependency_edges_by_llm

        monkeypatch.setenv("TOPO_DAG_LLM_REFINE", "1")
        steps = self._parsed_steps(
            "Step 1: Let x=1.\nStep 2: Then y=x+1.\nStep 3: Conclude y=2."
        )

        class CrashClient:
            def generate(self, prompt: str) -> str:
                raise RuntimeError("backend exploded")

        edges, _ = build_dependency_edges_by_llm(steps, llm_client=CrashClient())
        assert edges  # rule edges still present
        assert all(not dep.startswith("llm_") for _s, _t, _et, dep in edges)


class TestBenchmarkSmoke:
    @pytest.mark.parametrize(
        "bench",
        [
            "gsm8k",
            "math500",
            "olympiadbench",
            "omni_math",
            "aime2024",
            "aime2025",
            "cnmo2024",
            "mmlu",
            "gpqa_diamond",
        ],
    )
    def test_fixture_trace_builds_valid_dag(self, bench):
        fixture = FIXTURE_DIR / f"{bench}.txt"
        text = fixture.read_text(encoding="utf-8")
        dag, _ = parse_answer_to_dag_debug(text, problem_id=f"smoke_{bench}")
        assert dag.num_nodes >= 3
        assert dag.is_valid_dag()
        direction = dag.direction_consistency()
        q_topo_proxy = 0.5 * float(direction) + 0.5 * float(1.0 if dag.num_edges > 0 else 0.0)
        assert q_topo_proxy > 0.0


class TestExtractorFlags:
    def test_new_flags_noop_when_unset(self, monkeypatch):
        # Explicitly clear new flags to ensure legacy path remains stable.
        for key in [
            "TOPO_DAG_EXTRA_STEP_MARKERS",
            "TOPO_DAG_LATEX_EXPR",
            "TOPO_DAG_BARRIER_STRICT",
            "TOPO_DAG_BARRIER_MIN_OVERLAP",
        ]:
            monkeypatch.delenv(key, raising=False)

        answer = "Step 1: let x=1\nStep 2: y=x+1\nStep 3: therefore y=2"
        dag_default, dbg_default = parse_answer_to_dag_debug(answer, problem_id="flag_default")

        # Keep flags disabled explicitly; output should stay unchanged.
        monkeypatch.setenv("TOPO_DAG_EXTRA_STEP_MARKERS", "0")
        monkeypatch.setenv("TOPO_DAG_LATEX_EXPR", "0")
        monkeypatch.setenv("TOPO_DAG_BARRIER_STRICT", "0")
        monkeypatch.setenv("TOPO_DAG_BARRIER_MIN_OVERLAP", "0.2")
        dag_off, dbg_off = parse_answer_to_dag_debug(answer, problem_id="flag_off")

        assert dag_default.num_nodes == dag_off.num_nodes
        assert dag_default.num_edges == dag_off.num_edges
        dep_default = sorted(d.get("dep_type", "") for _, _, d in dag_default.graph.edges(data=True))
        dep_off = sorted(d.get("dep_type", "") for _, _, d in dag_off.graph.edges(data=True))
        assert dep_default == dep_off
        assert dbg_default["summary"]["num_steps"] == dbg_off["summary"]["num_steps"]
