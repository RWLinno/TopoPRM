from __future__ import annotations

from typing import Any

from swift.rewards import ORM, orms

from src.reward.outcome_reward import OutcomeReward
from src.reward.format_reward import FormatReward
from src.reward.topo_reward import TopoReward
from src.reward.continuity_reward import ContinuityReward


class LengthReward(ORM):
    """Length-penalty reward.

    * ≤ 2000 characters → **1.0**
    * 2000–4000 characters → linear decay from 1.0 to 0.0
    * > 4000 characters → **0.0**
    """

    LOW: int = 2000
    HIGH: int = 4000

    def __call__(
        self,
        completions: list,
        **kwargs: Any,
    ) -> list[float]:
        rewards: list[float] = []
        for completion in completions:
            text = completion if isinstance(completion, str) else (completion[-1].get("content", "") if completion else "")
            length = len(text)
            if length <= self.LOW:
                rewards.append(1.0)
            elif length >= self.HIGH:
                rewards.append(0.0)
            else:
                rewards.append(1.0 - (length - self.LOW) / (self.HIGH - self.LOW))
        return rewards


class TopoCompositeReward(ORM):
    """Weighted composite reward.

    ``R = 0.4×outcome + 0.15×format + 0.2×topo + 0.15×continuity + 0.1×length``
    """

    WEIGHTS: dict[str, float] = {
        "outcome": 0.40,
        "format": 0.15,
        "topo": 0.20,
        "continuity": 0.15,
        "length": 0.10,
    }

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self._outcome = OutcomeReward()
        self._format = FormatReward()
        self._topo = TopoReward()
        self._continuity = ContinuityReward()
        self._length = LengthReward()

    def __call__(
        self,
        completions: list,
        solution: Any = None,
        reference_dag: Any = None,
        **kwargs: Any,
    ) -> list[float]:
        outcome_scores = self._outcome(completions, solution=solution, **kwargs)
        format_scores = self._format(completions, **kwargs)
        topo_scores = self._topo(completions, reference_dag=reference_dag, **kwargs)
        continuity_scores = self._continuity(completions, **kwargs)
        length_scores = self._length(completions, **kwargs)

        w = self.WEIGHTS
        rewards: list[float] = []
        for o, f, t, c, l in zip(
            outcome_scores, format_scores, topo_scores, continuity_scores, length_scores
        ):
            r = (
                w["outcome"] * o
                + w["format"] * f
                + w["topo"] * t
                + w["continuity"] * c
                + w["length"] * l
            )
            rewards.append(round(r, 6))
        return rewards


def get_reward_func(reward_type: str = "composite") -> ORM:
    """Return a reward class instance by name.

    Parameters
    ----------
    reward_type:
        One of ``"composite"``, ``"outcome"``, ``"format"``, ``"topo"``,
        ``"continuity"``, ``"length"``.

    Returns
    -------
    ORM
        The corresponding reward callable.
    """
    registry: dict[str, type[ORM]] = {
        "composite": TopoCompositeReward,
        "outcome": OutcomeReward,
        "format": FormatReward,
        "topo": TopoReward,
        "continuity": ContinuityReward,
        "length": LengthReward,
    }
    if reward_type not in registry:
        raise ValueError(
            f"Unknown reward_type {reward_type!r}. "
            f"Choose from {sorted(registry.keys())}."
        )
    return registry[reward_type]()


# ---------------------------------------------------------------------------
# Register all reward classes in the global ``orms`` dict so that SWIFT's
# plugin system can discover them by name.
# ---------------------------------------------------------------------------
orms["topo_composite"] = TopoCompositeReward
orms["topo_outcome"] = OutcomeReward
orms["topo_format"] = FormatReward
orms["topo_topo"] = TopoReward
orms["topo_continuity"] = ContinuityReward
orms["topo_length"] = LengthReward
