from __future__ import annotations

import json
import re
from typing import Any, Optional

from swift.rewards import ORM, orms
from src.reward.utils import completion_to_text

try:
    from math_verify import LatexExtractionConfig, parse, verify
    _MATH_VERIFY = True
except ImportError:
    _MATH_VERIFY = False


class OutcomeReward(ORM):
    """Unified outcome reward supporting both math-benchmark and critique tasks.

    For public math benchmarks (solution is a simple numeric/latex answer):
      - Extracts \\boxed{} from model output and uses math_verify for equivalence.
      - Falls back to simple string matching of last numeric value.

    For critique/grading tasks (solution is a JSON with 学生得分/结论批改):
      - Parses <answer> JSON block as before.
    """

    MAX_RAW: float = 1.5

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_answer_json(text: str) -> Optional[dict[str, Any]]:
        """Return the first JSON object found inside ``<answer>…</answer>``."""
        m = re.search(r"<answer>\s*(.*?)\s*</answer>", text, re.DOTALL)
        if m is None:
            return None
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            return None

    @staticmethod
    def _get_score(obj: dict[str, Any]) -> Optional[float]:
        """Extract numeric score from parsed answer JSON."""
        for key in ("学生得分", "score", "得分"):
            if key in obj:
                try:
                    return float(obj[key])
                except (TypeError, ValueError):
                    continue
        return None

    @staticmethod
    def _get_conclusion(obj: dict[str, Any]) -> Optional[str]:
        """Extract critique conclusion string."""
        for key in ("结论批改", "conclusion", "批改结论"):
            if key in obj:
                return str(obj[key]).strip()
        return None

    @staticmethod
    def _parse_solution(solution: Any) -> tuple[Optional[float], Optional[str]]:
        """Parse ground truth *solution* into (score, conclusion)."""
        if solution is None:
            return None, None
        if isinstance(solution, (int, float)):
            return float(solution), None
        if isinstance(solution, str):
            try:
                solution = json.loads(solution)
            except json.JSONDecodeError:
                try:
                    return float(solution), None
                except ValueError:
                    return None, None
        if isinstance(solution, dict):
            gt_score: Optional[float] = None
            for key in ("学生得分", "score", "得分"):
                if key in solution:
                    try:
                        gt_score = float(solution[key])
                    except (TypeError, ValueError):
                        continue
                    break
            gt_conclusion: Optional[str] = None
            for key in ("结论批改", "conclusion", "批改结论"):
                if key in solution:
                    gt_conclusion = str(solution[key]).strip()
                    break
            return gt_score, gt_conclusion
        return None, None

    # ------------------------------------------------------------------
    # main
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # math benchmark helpers
    # ------------------------------------------------------------------

    _BOXED_RE = re.compile(r"\\boxed\{([^}]*)\}")
    _LAST_NUM_RE = re.compile(r"([-+]?\d*\.?\d+)\s*$")

    @classmethod
    def _extract_math_answer(cls, text: str) -> Optional[str]:
        """Extract answer from \\boxed{} or last number in text."""
        matches = cls._BOXED_RE.findall(text)
        if matches:
            return matches[-1].strip()
        m = cls._LAST_NUM_RE.search(text)
        if m:
            return m.group(1)
        return None

    @classmethod
    def _math_equiv(cls, prediction: str, ground_truth: str) -> bool:
        """Check mathematical equivalence using math_verify if available."""
        if _MATH_VERIFY:
            try:
                parsed_pred = parse(prediction, extraction_config=[
                    LatexExtractionConfig(boxed=True, boxed_match_priority=0),
                    LatexExtractionConfig()])
                parsed_gt = parse(ground_truth, extraction_config=[
                    LatexExtractionConfig(boxed=True, boxed_match_priority=0),
                    LatexExtractionConfig()])
                return verify(parsed_pred, parsed_gt)
            except Exception:
                pass
        pred_clean = prediction.strip().rstrip('.').strip()
        gt_clean = ground_truth.strip().rstrip('.').strip()
        return pred_clean == gt_clean

    def _is_critique_solution(self, sol: Any) -> bool:
        """Detect if solution is for critique task (JSON with scores)."""
        if isinstance(sol, dict):
            return any(k in sol for k in ("学生得分", "score", "结论批改"))
        if isinstance(sol, str):
            try:
                d = json.loads(sol)
                return isinstance(d, dict) and any(k in d for k in ("学生得分", "score", "结论批改"))
            except (json.JSONDecodeError, ValueError):
                pass
        return False

    # ------------------------------------------------------------------
    # main
    # ------------------------------------------------------------------

    def __call__(
        self,
        completions: list,
        solution: Any = None,
        **kwargs: Any,
    ) -> list[float]:
        """Return a reward in [0, 1] for each completion."""
        solutions = solution if isinstance(solution, list) else [solution] * len(completions)

        rewards: list[float] = []
        for i, completion in enumerate(completions):
            text = completion_to_text(completion)
            sol_i = solutions[i] if i < len(solutions) else None

            # Route: critique task vs math benchmark
            if sol_i is not None and self._is_critique_solution(sol_i):
                # Critique grading path (original logic)
                gt_score, gt_conclusion = self._parse_solution(sol_i)
                ans = self._extract_answer_json(text)
                if ans is None or gt_score is None:
                    rewards.append(0.0)
                    continue
                raw = 0.0
                pred_score = self._get_score(ans)
                if pred_score is not None:
                    if pred_score == gt_score:
                        raw += 1.0
                    elif abs(pred_score - gt_score) <= 1.0:
                        raw += 0.5
                if gt_conclusion is not None:
                    pred_conclusion = self._get_conclusion(ans)
                    if pred_conclusion is not None and pred_conclusion == gt_conclusion:
                        raw += 0.5
                raw = min(raw, self.MAX_RAW)
                rewards.append(raw / self.MAX_RAW)
            else:
                # Math benchmark path: extract boxed answer and verify
                if sol_i is None:
                    rewards.append(0.0)
                    continue
                gt_str = str(sol_i).strip()
                pred = self._extract_math_answer(text)
                if pred is None:
                    rewards.append(0.0)
                else:
                    rewards.append(1.0 if self._math_equiv(pred, gt_str) else 0.0)

        return rewards
