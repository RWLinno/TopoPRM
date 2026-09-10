from dataclasses import replace

from src.distill.build_srt_data import (
    DistillationRecord,
    build_prompt_dispatch,
    build_revision_instruction,
    derive_record_seed,
    revision_rejection_reason,
)


def test_process_aware_filter_keeps_high_quality() -> None:
    from src.distill.teacher_trace_filter import TraceFilterConfig, process_aware_trace_filter

    responses = [
        "<think>∵ x=1\n∴ y=x+1\n故 y=2</think><answer>{}</answer>",
        "<think></think><answer>{}</answer>",
    ]
    kept = process_aware_trace_filter(responses, TraceFilterConfig(min_quality=0.3))
    assert len(kept) >= 1
    assert kept[0][1] >= 0.3


def test_process_aware_filter_sorts_desc() -> None:
    from src.distill.teacher_trace_filter import TraceFilterConfig, process_aware_trace_filter

    responses = [
        "<think>∵ a=1\n∴ b=2\n故 c=3</think><answer>{}</answer>",
        "<think>短</think><answer>{}</answer>",
    ]
    kept = process_aware_trace_filter(responses, TraceFilterConfig(min_quality=0.0))
    assert len(kept) == 2
    assert kept[0][1] >= kept[1][1]


def _accepted_revision() -> DistillationRecord:
    return DistillationRecord(
        problem="p",
        solution="s",
        y_init="initial",
        P_r="revise",
        y_revised="<think>supported</think>\\boxed{1}",
        defect_type="cycle",
        r_out_init=0,
        r_out_revised=1,
        q_topo_init=0.5,
        q_topo_revised=0.8,
        q_dir_init=0.6,
        q_dir_revised=0.7,
        q_acyc_init=0.7,
        q_acyc_revised=0.8,
        q_cont_init=0.5,
        q_cont_revised=0.8,
        revised_tokens=128,
        format_ok_revised=True,
    )


def test_topology_dispatch_prioritizes_cycle_and_preserves_answer_warning() -> None:
    kind, prompt = build_prompt_dispatch(
        0,
        defect={
            "cycle_components": [[2, 4]],
            "backward_edges": [{"source": 4, "target": 2}],
        },
    )
    assert kind == "cycle"
    assert "steps 2, 4" in prompt
    assert prompt.startswith("The final answer is also incorrect.")


def test_topology_dispatch_localizes_backward_edge() -> None:
    kind, prompt = build_prompt_dispatch(
        1,
        defect={"backward_edges": [{"source": 5, "target": 3}]},
    )
    assert kind == "backward"
    assert "Step 5 depends on later step 3" in prompt


def test_revision_contract_rejects_structural_regression_and_over_budget() -> None:
    record = _accepted_revision()
    assert revision_rejection_reason(record, topo_threshold=0.5, token_budget=1024) == ""
    assert revision_rejection_reason(
        replace(record, q_dir_revised=0.4), topo_threshold=0.5, token_budget=1024
    ) == "direction_degraded"
    assert revision_rejection_reason(
        replace(record, q_acyc_revised=0.4), topo_threshold=0.5, token_budget=1024
    ) == "acyclicity_degraded"
    assert revision_rejection_reason(
        replace(record, revised_tokens=1025), topo_threshold=0.5, token_budget=1024
    ) == "over_budget"


def test_basic_control_selection_uses_only_answer_format_and_budget() -> None:
    record = replace(
        _accepted_revision(),
        q_topo_revised=0.1,
        q_dir_revised=0.1,
        q_acyc_revised=0.1,
    )
    assert revision_rejection_reason(
        record,
        topo_threshold=0.5,
        token_budget=1024,
        selection="basic",
    ) == ""


def test_revision_strategies_are_explicit_and_budget_matched() -> None:
    score = {
        "r_out": 1,
        "r_topo": 0.4,
        "defect": {"cycle_components": [[1, 3]]},
    }
    assert build_revision_instruction(
        "topology", score=score, token_budget=1024
    )[0] == "cycle"
    assert build_revision_instruction(
        "generic", score=score, token_budget=1024
    )[0] == "generic"
    kind, prompt = build_revision_instruction("length", score=score, token_budget=777)
    assert kind == "length"
    assert "777 tokens" in prompt
    assert build_revision_instruction(
        "static", score=score, token_budget=1024
    ) == ("static", "")
    assert derive_record_seed(0, "problem-1:sample-0", "student") == derive_record_seed(
        0, "problem-1:sample-0", "student"
    )
    assert derive_record_seed(0, "problem-1:sample-0", "student") != derive_record_seed(
        0, "problem-2:sample-0", "student"
    )
