from __future__ import annotations

import re
from typing import Any

from swift.rewards import ORM, orms
from src.reward.utils import completion_to_text


class FormatReward(ORM):
    """Format-compliance reward for math reasoning benchmarks.

    Checks whether the model output follows a recognizable reasoning format:
      - <think>...</think> followed by a final answer (\\boxed{} or explicit statement)
      - Or at least contains structured step-by-step reasoning with a boxed answer.

    Scoring rubric (progressive):
      1.0 — has <think> block AND \\boxed{} answer
      0.8 — has \\boxed{} answer with multi-step reasoning (no explicit <think>)
      0.5 — has \\boxed{} answer only (minimal reasoning)
      0.3 — has <think> but no \\boxed{} (reasoning without conclusion)
      0.1 — has some step-by-step structure but no boxed answer
      0.0 — unstructured or empty
    """

    _THINK_RE = re.compile(r"<think>", re.IGNORECASE)
    _THINK_CLOSE_RE = re.compile(r"</think>", re.IGNORECASE)
    _BOXED_RE = re.compile(r"\\boxed\{")
    _STEP_MARKERS = re.compile(
        r"(step\s*\d|first|second|third|therefore|thus|hence|so\s+the|"
        r"we\s+(get|have|find|know|can)|let\s+|since\s+|because\s+)",
        re.IGNORECASE,
    )

    def __call__(self, completions: list, **kwargs: Any) -> list[float]:
        rewards: list[float] = []
        for completion in completions:
            text = completion_to_text(completion)
            if not text.strip():
                rewards.append(0.0)
                continue

            has_think = bool(self._THINK_RE.search(text))
            has_boxed = bool(self._BOXED_RE.search(text))
            has_steps = len(self._STEP_MARKERS.findall(text)) >= 2

            if has_think and has_boxed:
                rewards.append(1.0)
            elif has_boxed and has_steps:
                rewards.append(0.8)
            elif has_boxed:
                rewards.append(0.5)
            elif has_think:
                rewards.append(0.3)
            elif has_steps:
                rewards.append(0.1)
            else:
                rewards.append(0.0)
        return rewards
