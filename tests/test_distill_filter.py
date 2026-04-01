from src.distill.teacher_trace_filter import TraceFilterConfig, process_aware_trace_filter


def test_process_aware_filter_keeps_high_quality() -> None:
    responses = [
        "<think>∵ x=1\n∴ y=x+1\n故 y=2</think><answer>{}</answer>",
        "<think></think><answer>{}</answer>",
    ]
    kept = process_aware_trace_filter(responses, TraceFilterConfig(min_quality=0.3))
    assert len(kept) >= 1
    assert kept[0][1] >= 0.3


def test_process_aware_filter_sorts_desc() -> None:
    responses = [
        "<think>∵ a=1\n∴ b=2\n故 c=3</think><answer>{}</answer>",
        "<think>短</think><answer>{}</answer>",
    ]
    kept = process_aware_trace_filter(responses, TraceFilterConfig(min_quality=0.0))
    assert len(kept) == 2
    assert kept[0][1] >= kept[1][1]
