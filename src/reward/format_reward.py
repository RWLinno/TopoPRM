from __future__ import annotations

import json
import re
from typing import Any

from swift.rewards import ORM, orms


class FormatReward(ORM):
    """Format-compliance reward.

    Checks whether the model output follows the expected
    ``<think>…</think><answer>…</answer>`` structure with a valid JSON
    payload inside ``<answer>``.

    Scoring rubric
    --------------
    * ``<think>`` present **and** ``<answer>`` with valid JSON → **1.0**
    * ``<answer>`` with valid JSON (but no ``<think>``) → **0.3**
    * Anything else → **0.0**
    """

    @staticmethod
    def _has_think(text: str) -> bool:
        return bool(re.search(r"<think>.*?</think>", text, re.DOTALL))

    @staticmethod
    def _has_answer_json(text: str) -> bool:
        m = re.search(r"<answer>\s*(.*?)\s*</answer>", text, re.DOTALL)
        if m is None:
            return False
        try:
            json.loads(m.group(1))
            return True
        except json.JSONDecodeError:
            return False

    def __call__(
        self,
        completions: list,
        **kwargs: Any,
    ) -> list[float]:
        """Return a format-compliance reward in {0.0, 0.3, 1.0} per completion."""
        rewards: list[float] = []
        for completion in completions:
            text = completion if isinstance(completion, str) else (completion[-1].get("content", "") if completion else "")
            has_answer = self._has_answer_json(text)
            if not has_answer:
                rewards.append(0.0)
            elif self._has_think(text):
                rewards.append(1.0)
            else:
                rewards.append(0.3)
        return rewards
