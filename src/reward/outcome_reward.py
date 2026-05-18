from __future__ import annotations

import re
from typing import Any, Optional

from math_verify import LatexExtractionConfig, parse, verify
from swift.rewards import ORM, orms
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

    _BOXED_RE = re.compile(r"\\boxed\{([^}]*(?:\{[^}]*\}[^}]*)*)\}")
    _LAST_NUM_RE = re.compile(r"(?:=\s*|is\s+|answer\s+is\s+)([-+]?\d*\.?\d+)")
    _HASH_ANS_RE = re.compile(r"####\s*([^\n]+)")
    _PLAIN_NUM_RE = re.compile(r"([-+]?\d+\.?\d*)\s*$")

    @classmethod
    def _extract_answer(cls, text: str) -> Optional[str]:
        """Extract the model's final answer from completion text."""
        # Priority 1: \\boxed{}
        matches = cls._BOXED_RE.findall(text)
        if matches:
            return matches[-1].strip()
        # Priority 2: "= X" or "answer is X" pattern
        matches = cls._LAST_NUM_RE.findall(text)
        if matches:
            return matches[-1].strip()
        # Priority 3: last standalone number
        matches = cls._PLAIN_NUM_RE.findall(text)
        if matches:
            return matches[-1].strip()
        return None

    @classmethod
    def _extract_ground_truth(cls, value: Any) -> Optional[str]:
        """Extract comparable GT answer from mixed dataset fields."""
        if value is None:
            return None
        if isinstance(value, dict):
            # Prefer explicit short answer when present.
            for key in ("final_answer", "answer", "solution", "standard_answer"):
                gt = cls._extract_ground_truth(value.get(key))
                if gt:
                    return gt
            return None
        if isinstance(value, list):
            for item in value:
                gt = cls._extract_ground_truth(item)
                if gt:
                    return gt
            return None

        text = str(value).strip()
        if not text:
            return None
        boxed = cls._extract_answer(text)
        if boxed:
            return boxed
        hash_ans = cls._HASH_ANS_RE.findall(text)
        if hash_ans:
            return hash_ans[-1].strip()
        return text

    @staticmethod
    def _verify_equivalence(prediction: str, ground_truth: str) -> bool:
        """Check mathematical equivalence using math_verify."""
        config = LatexExtractionConfig(boxed_match_priority=0)
        parsed_pred = parse(prediction, extraction_config=[config])
        parsed_gt = parse(ground_truth, extraction_config=[config])
        if parsed_pred and parsed_gt:
            return verify(parsed_pred, parsed_gt)
        # Fallback: direct string comparison after normalization
        pred_clean = prediction.strip().rstrip(".").strip()
        gt_clean = ground_truth.strip().rstrip(".").strip()
        return pred_clean == gt_clean

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
            gt_str = self._extract_ground_truth(gt)
            if not gt_str:
                rewards.append(0.0)
                continue

            pred = self._extract_answer(text)
            if pred is None:
                rewards.append(0.0)
                continue

            is_correct = self._verify_equivalence(pred, gt_str)
            rewards.append(1.0 if is_correct else 0.0)

        return rewards
