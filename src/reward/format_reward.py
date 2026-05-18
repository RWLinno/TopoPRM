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
    _THINK_BLOCK_RE = re.compile(r"<think>(.*?)</think>", re.IGNORECASE | re.DOTALL)
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
            has_think_close = bool(self._THINK_CLOSE_RE.search(text))
            has_boxed = bool(self._BOXED_RE.search(text))
            has_steps = len(self._STEP_MARKERS.findall(text)) >= 2
            think_match = self._THINK_BLOCK_RE.search(text)
            think_tokens = len(think_match.group(1).split()) if think_match else 0
            total_tokens = len(text.split())
            chain_tokens = max(think_tokens, total_tokens)

            # Progressive long-CoT shaping:
            # - Prefer well-formed <think>...</think> + boxed answer.
            # - Give extra gain to longer chains to support AIME-like traces.
            if has_think and has_think_close and has_boxed:
                chain_bonus = min(0.2, chain_tokens / 2000.0)
                step_bonus = 0.05 if has_steps else 0.0
                rewards.append(min(1.0, 0.75 + chain_bonus + step_bonus))
            elif has_think and has_boxed:
                chain_bonus = min(0.15, chain_tokens / 2200.0)
                rewards.append(min(0.95, 0.70 + chain_bonus))
            elif has_boxed and has_steps:
                chain_bonus = min(0.2, total_tokens / 1800.0)
                rewards.append(min(0.85, 0.55 + chain_bonus))
            elif has_boxed:
                rewards.append(0.5)
            elif has_think:
                rewards.append(0.35 if has_think_close else 0.25)
            elif has_steps:
                rewards.append(0.1)
            else:
                rewards.append(0.0)
        return rewards
