from __future__ import annotations

import re
from typing import Any, Optional

from swift.rewards import ORM, orms
from src.eval.math_scoring import (
    extract_last_boxed,
    parse_answer_candidate,
    verify_answer_equivalence,
    verify_math_response,
)
from src.reward.utils import completion_to_text


class OutcomeReward(ORM):
    """Math outcome reward: verifies if model answer matches ground truth.

    Extraction priority:
      1. \\boxed{...} — standard LaTeX boxed answer
      2. Last numeric value in the response (fallback for informal answers)

    Verification uses math_verify for symbolic equivalence (handles
    different representations of the same number/expression).

    Returns 1.0 for correct, 0.0 for incorrect or unparseable.
    """

    _BOXED_START_RE = re.compile(r"\\boxed\{")
    _LAST_NUM_RE = re.compile(r"(?:=\s*|is\s+|answer\s+is\s+)([-+]?\d*\.?\d+)")
    _PLAIN_NUM_RE = re.compile(r"([-+]?\d+\.?\d*)\s*$")

    @classmethod
    def _extract_last_boxed(cls, text: str) -> Optional[str]:
        return extract_last_boxed(text)

    @staticmethod
    def _parse_answer_candidate(value: str) -> list:
        """Parse a known final-answer candidate as one complete math object."""
        return parse_answer_candidate(value)

    @classmethod
    def _extract_answer(cls, text: str) -> Optional[str]:
        """Extract the model's final answer from completion text."""
        # Priority 1: \\boxed{}
        boxed = cls._extract_last_boxed(text)
        if boxed is not None:
            return boxed
        # Priority 2: "= X" or "answer is X" pattern
        matches = cls._LAST_NUM_RE.findall(text)
        if matches:
            return matches[-1].strip()
        # Priority 3: last standalone number
        matches = cls._PLAIN_NUM_RE.findall(text)
        if matches:
            return matches[-1].strip()
        return None

    @staticmethod
    def _verify_equivalence(prediction: str, ground_truth: str) -> bool:
        """Check mathematical equivalence using math_verify."""
        return verify_answer_equivalence(prediction, ground_truth)

    @staticmethod
    def verify_math_response(response: str, ground_truth: str) -> bool:
        """Score the final non-empty box or an explicit final-answer region."""
        return verify_math_response(response, ground_truth)

    def __call__(
        self,
        completions: list,
        solution: Any = None,
        **kwargs: Any,
    ) -> list[float]:
        """Return 1.0 for correct answer, 0.0 otherwise."""
        if solution is None:
            return [0.0] * len(completions)

        solutions = solution if isinstance(solution, list) else [solution] * len(completions)
        rewards: list[float] = []

        for i, completion in enumerate(completions):
            text = completion_to_text(completion)
            gt = solutions[i] if i < len(solutions) else None
            if gt is None or str(gt).strip() == "":
                rewards.append(0.0)
                continue

            gt_str = str(gt).strip()
            pred = self._extract_answer(text)
            if pred is None:
                rewards.append(0.0)
                continue

            is_correct = self._verify_equivalence(pred, gt_str)
            rewards.append(1.0 if is_correct else 0.0)

        return rewards
