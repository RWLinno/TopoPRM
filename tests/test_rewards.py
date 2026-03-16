import json

import pytest

from src.reward.outcome_reward import OutcomeReward
from src.reward.format_reward import FormatReward
from src.reward.topo_reward import TopoReward
from src.reward.continuity_reward import ContinuityReward
from src.reward.composite_reward import LengthReward, TopoCompositeReward


def _wrap(text: str):
    """Wrap text into the ``completions`` format expected by reward callables."""
    return [[{"content": text}]]


# ---------------------------------------------------------------------------
# OutcomeReward
# ---------------------------------------------------------------------------

class TestOutcomeReward:
    def setup_method(self):
        self.reward = OutcomeReward()

    def test_exact_match(self):
        text = '<think>分析</think><answer>{"学生得分": 5, "结论批改": "正确"}</answer>'
        solution = json.dumps({"学生得分": 5, "结论批改": "正确"})
        scores = self.reward(_wrap(text), solution=solution)
        assert scores == [1.0]

    def test_close_match(self):
        text = '<think>分析</think><answer>{"学生得分": 4}</answer>'
        solution = json.dumps({"学生得分": 5})
        scores = self.reward(_wrap(text), solution=solution)
        assert scores[0] == pytest.approx(0.5 / 1.5, abs=1e-6)

    def test_no_match(self):
        text = '<think>分析</think><answer>{"学生得分": 0}</answer>'
        solution = json.dumps({"学生得分": 5})
        scores = self.reward(_wrap(text), solution=solution)
        assert scores == [0.0]

    def test_missing_solution(self):
        text = '<think>分析</think><answer>{"学生得分": 5}</answer>'
        scores = self.reward(_wrap(text), solution=None)
        assert scores == [0.0]

    def test_multiple_completions(self):
        completions = [
            [{"content": '<answer>{"学生得分": 5}</answer>'}],
            [{"content": '<answer>{"学生得分": 3}</answer>'}],
        ]
        solution = json.dumps({"学生得分": 5})
        scores = self.reward(completions, solution=solution)
        assert len(scores) == 2
        assert scores[0] > scores[1]


# ---------------------------------------------------------------------------
# FormatReward
# ---------------------------------------------------------------------------

class TestFormatReward:
    def setup_method(self):
        self.reward = FormatReward()

    def test_perfect_format(self):
        text = '<think>推理过程</think><answer>{"score": 5}</answer>'
        assert self.reward(_wrap(text)) == [1.0]

    def test_partial_no_think(self):
        text = '<answer>{"score": 5}</answer>'
        assert self.reward(_wrap(text)) == [0.3]

    def test_missing_tags(self):
        text = "这是一段普通文本，没有任何标签"
        assert self.reward(_wrap(text)) == [0.0]

    def test_invalid_answer_json(self):
        text = "<think>ok</think><answer>NOT VALID JSON</answer>"
        assert self.reward(_wrap(text)) == [0.0]


# ---------------------------------------------------------------------------
# TopoReward
# ---------------------------------------------------------------------------

class TestTopoReward:
    def setup_method(self):
        self.reward = TopoReward()

    def test_good_reasoning(self):
        think = "∵ x = 1\n∴ y = x + 1\n故 y = 2"
        text = f"<think>{think}</think><answer>ok</answer>"
        scores = self.reward(_wrap(text))
        assert scores[0] > 0.0

    def test_empty_think(self):
        text = "<think></think><answer>ok</answer>"
        assert self.reward(_wrap(text)) == [0.0]

    def test_no_tags(self):
        text = "普通文本没有think标签"
        assert self.reward(_wrap(text)) == [0.0]

    def test_acyclic_bonus(self):
        think = "∵ a = 1\n∴ b = 2\n故 c = 3"
        text = f"<think>{think}</think><answer>ok</answer>"
        scores = self.reward(_wrap(text))
        assert scores[0] >= 0.6  # base 0.4 + acyclic 0.2


# ---------------------------------------------------------------------------
# ContinuityReward
# ---------------------------------------------------------------------------

class TestContinuityReward:
    def setup_method(self):
        self.reward = ContinuityReward()

    def test_no_think_block(self):
        text = "没有think标签的文本"
        assert self.reward(_wrap(text)) == [0.0]

    def test_empty_think(self):
        text = "<think></think>"
        assert self.reward(_wrap(text)) == [0.0]

    def test_continuous_reasoning(self):
        think = "已知 $x=1$\n由题意 $y=x+1=2$\n故 答案为2"
        text = f"<think>{think}</think>"
        scores = self.reward(_wrap(text))
        assert len(scores) == 1

    def test_single_step(self):
        text = "<think>已知 x = 1</think>"
        scores = self.reward(_wrap(text))
        assert len(scores) == 1


# ---------------------------------------------------------------------------
# LengthReward
# ---------------------------------------------------------------------------

class TestLengthReward:
    def setup_method(self):
        self.reward = LengthReward()

    def test_ideal_length(self):
        text = "短文本" * 10
        assert self.reward(_wrap(text)) == [1.0]

    def test_too_long(self):
        text = "x" * 5000
        assert self.reward(_wrap(text)) == [0.0]

    def test_boundary_low(self):
        text = "x" * 2000
        assert self.reward(_wrap(text)) == [1.0]

    def test_boundary_high(self):
        text = "x" * 4000
        assert self.reward(_wrap(text)) == [0.0]

    def test_linear_decay(self):
        text = "x" * 3000
        scores = self.reward(_wrap(text))
        assert scores[0] == pytest.approx(0.5, abs=1e-6)


# ---------------------------------------------------------------------------
# TopoCompositeReward
# ---------------------------------------------------------------------------

class TestTopoCompositeReward:
    def setup_method(self):
        self.reward = TopoCompositeReward()

    def test_output_length(self):
        completions = [
            [{"content": "text1"}],
            [{"content": "text2"}],
            [{"content": "text3"}],
        ]
        scores = self.reward(completions, solution=None)
        assert len(scores) == 3

    def test_values_in_range(self):
        text = '<think>∵ x=1\n∴ y=2</think><answer>{"学生得分": 5}</answer>'
        solution = json.dumps({"学生得分": 5})
        scores = self.reward(_wrap(text), solution=solution)
        assert all(0.0 <= s <= 1.0 for s in scores)

    def test_weights_sum_to_one(self):
        assert sum(TopoCompositeReward.WEIGHTS.values()) == pytest.approx(1.0)
