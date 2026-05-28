from __future__ import annotations

import json
import re
from typing import Any

from swift.rewards import ORM, orms
from src.reward.utils import completion_to_text


class FormatReward(ORM):
    """Format-compliance reward.

    Checks whether the model output follows the expected
    ``<think>…</think><answer>…</answer>`` structure with a valid JSON
    payload inside ``<answer>``.

    Scoring rubric (progressive curriculum)
    --------------
    * ``<think>`` present **and** ``<answer>`` with valid JSON → **1.0**
    * ``<answer>`` with valid JSON (but no ``<think>``) → **0.5**
    * ``<think>`` present but no ``<answer>`` (e.g. truncated) → **0.1**
    * Anything else → **0.0**

    The 0.1 partial credit for ``<think>``-only outputs provides a
    curriculum signal: the model first learns to enter reasoning mode,
    then learns to produce a parseable answer block.
    """

    @staticmethod
    def _has_think(text: str) -> bool:
        return bool(re.search(r"<think>", text))

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
        """Return a format-compliance reward per completion."""
        rewards: list[float] = []
        for completion in completions:
            text = completion_to_text(completion)
            has_answer = self._has_answer_json(text)
            has_think = self._has_think(text)
            if has_answer and has_think:
                rewards.append(1.0)
            elif has_answer:
                rewards.append(0.5)
            elif has_think:
                rewards.append(0.1)
            else:
                rewards.append(0.0)
        return rewards
