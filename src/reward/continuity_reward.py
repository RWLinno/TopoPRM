from __future__ import annotations

import re
from typing import Any

from swift.rewards import ORM, orms

from src.data.build_dag import (
    canonicalize_expression,
    extract_claim_keys,
    extract_expressions,
    extract_steps_from_answer,
)
from src.reward.reward_config import RewardConfig
from src.reward.utils import completion_to_text, extract_think_block

_GIVEN_PATTERNS = re.compile(
    r"已知|given|由题意|题目|条件|根据题|由题目|题设",
    re.IGNORECASE,
)


class ContinuityReward(ORM):
    BROKEN_CHAIN_PENALTY: float = RewardConfig.CONTINUITY_BROKEN_CHAIN_PENALTY

    """Step-continuity reward.

    Each reasoning step's expressions and claims must be *traceable* to
    a prior step or to the given conditions.  Steps whose text contains
    markers such as ``已知``, ``given``, or ``由题意`` are treated as
    automatically continuous (they cite the problem statement).

    Scoring
    -------
    * ``score = num_continuous / total_steps``
    * If every step is continuous → **1.0**
    * Otherwise → ``score × 0.8`` (penalty for broken chains)
    """

    @staticmethod
    def _is_given_step(text: str) -> bool:
        return bool(_GIVEN_PATTERNS.search(text))

    def __call__(
        self,
        completions: list,
        **kwargs: Any,
    ) -> list[float]:
        """Return continuity reward in [0, 1] per completion."""
        rewards: list[float] = []
        for completion in completions:
            text = completion_to_text(completion)
            score, _ = self.diagnose(text)
            rewards.append(score)
        return rewards

    def diagnose(self, text: str) -> tuple[float, list[int]]:
        """Return the continuity score and localized unsupported step indices."""
        think_text = extract_think_block(text)

        steps = extract_steps_from_answer(think_text)
        if not steps:
            return 0.0, []

        prior_exprs: set[str] = set()
        prior_claims: set[str] = set()
        continuous_count = 0
        breaks: list[int] = []

        for index, step in enumerate(steps):
            step_text = step["raw_text"] if isinstance(step, dict) else str(step)
            cur_exprs = {canonicalize_expression(e) for e in extract_expressions(step_text)}
            cur_claims = set(extract_claim_keys(step_text))
            is_continuous = False

            if self._is_given_step(step_text):
                is_continuous = True
            elif not cur_exprs and not cur_claims:
                is_continuous = not RewardConfig.CONTINUITY_REQUIRE_EVIDENCE
            else:
                is_continuous = bool(cur_exprs & prior_exprs) or bool(cur_claims & prior_claims)

            if is_continuous:
                continuous_count += 1
            else:
                breaks.append(index)
            prior_exprs.update(cur_exprs)
            prior_claims.update(cur_claims)

        ratio = continuous_count / len(steps)
        score = 1.0 if ratio >= 1.0 else ratio * self.BROKEN_CHAIN_PENALTY
        return score, breaks
