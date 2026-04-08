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
            think_text = extract_think_block(text)

            steps = extract_steps_from_answer(think_text)
            if not steps:
                rewards.append(0.0)
                continue

            prior_exprs: set[str] = set()
            prior_claims: set[str] = set()
            continuous_count = 0

            for step in steps:
                step_text = step["raw_text"] if isinstance(step, dict) else str(step)
                cur_exprs = {canonicalize_expression(e) for e in extract_expressions(step_text)}
                # Use canonical claim keys for continuity matching.
                # Sentence-level claims are kept for display/debug, not matching.
                cur_claims = set(extract_claim_keys(step_text))

                if self._is_given_step(step_text):
                    continuous_count += 1
                elif not cur_exprs and not cur_claims:
                    continuous_count += 1
                else:
                    overlaps_expr = bool(cur_exprs & prior_exprs)
                    overlaps_claim = bool(cur_claims & prior_claims)
                    if overlaps_expr or overlaps_claim:
                        continuous_count += 1

                prior_exprs.update(cur_exprs)
                prior_claims.update(cur_claims)

            total = len(steps)
            ratio = continuous_count / total
            if ratio >= 1.0:
                rewards.append(1.0)
            else:
                rewards.append(ratio * self.BROKEN_CHAIN_PENALTY)

        return rewards
