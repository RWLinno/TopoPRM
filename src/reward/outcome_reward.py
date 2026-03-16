from __future__ import annotations

import json
import re
from typing import Any, Optional

from swift.rewards import ORM, orms


class OutcomeReward(ORM):
    """Score-accuracy reward for math critique / grading tasks.

    Parses the model's ``<answer>`` JSON block to extract the predicted
    student score (``学生得分`` or ``score``) and the critique conclusion
    (``结论批改``).  These are compared against the ground-truth solution
    supplied via the *solution* kwarg.

    Scoring rubric
    --------------
    * Exact score match: **+1.0**
    * Within ±1 of the ground-truth score: **+0.5**
    * ``结论批改`` fully matches the ground-truth conclusion: **+0.5** bonus
    * Raw total is capped at **1.5** and then normalised to **[0, 1]**.
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
            text = completion if isinstance(completion, str) else (completion[-1].get("content", "") if completion else "")
            sol_i = solutions[i] if i < len(solutions) else None
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

        return rewards
