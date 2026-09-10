"""Ablation reward configurations for NeurIPS experiments.

Each class uses a different subset/weighting of reward components.
All register in the ms-swift ``orms`` dict for use via --reward_funcs.
"""

from __future__ import annotations

from typing import Any, List

from swift.rewards import ORM, orms

from src.reward.continuity_reward import ContinuityReward
from src.reward.format_reward import FormatReward
from src.reward.outcome_reward import OutcomeReward
from src.reward.topo_reward import TopoReward
from src.reward.composite_reward import LengthReward


class OutcomeOnlyReward(ORM):
    """Ablation: outcome reward only (weight=1.0)."""

    def __init__(self, **kwargs) -> None:
        self._outcome = OutcomeReward()

    def __call__(self, completions: list, **kwargs: Any) -> List[float]:
        return self._outcome(completions, **kwargs)


class NoTopoReward(ORM):
    """Ablation: remove topology reward, redistribute weight to outcome."""

    WEIGHTS = {"outcome": 0.55, "format": 0.15, "continuity": 0.20, "length": 0.10}

    def __init__(self, **kwargs) -> None:
        self._outcome = OutcomeReward()
        self._format = FormatReward()
        self._continuity = ContinuityReward()
        self._length = LengthReward()

    def __call__(self, completions: list, **kwargs: Any) -> List[float]:
        r_out = self._outcome(completions, **kwargs)
        r_fmt = self._format(completions, **kwargs)
        r_con = self._continuity(completions, **kwargs)
        r_len = self._length(completions, **kwargs)
        w = self.WEIGHTS
        return [
            round(w["outcome"] * a + w["format"] * b + w["continuity"] * c + w["length"] * d, 4)
            for a, b, c, d in zip(r_out, r_fmt, r_con, r_len)
        ]


class NoContinuityReward(ORM):
    """Ablation: remove continuity reward, redistribute weight to outcome."""

    WEIGHTS = {"outcome": 0.55, "format": 0.15, "topo": 0.20, "length": 0.10}

    def __init__(self, **kwargs) -> None:
        self._outcome = OutcomeReward()
        self._format = FormatReward()
        self._topo = TopoReward()
        self._length = LengthReward()

    def __call__(self, completions: list, **kwargs: Any) -> List[float]:
        r_out = self._outcome(completions, **kwargs)
        r_fmt = self._format(completions, **kwargs)
        r_top = self._topo(completions, **kwargs)
        r_len = self._length(completions, **kwargs)
        w = self.WEIGHTS
        return [
            round(w["outcome"] * a + w["format"] * b + w["topo"] * c + w["length"] * d, 4)
            for a, b, c, d in zip(r_out, r_fmt, r_top, r_len)
        ]


class NoFormatReward(ORM):
    """Ablation: remove format reward, redistribute weight."""

    WEIGHTS = {"outcome": 0.45, "topo": 0.25, "continuity": 0.20, "length": 0.10}

    def __init__(self, **kwargs) -> None:
        self._outcome = OutcomeReward()
        self._topo = TopoReward()
        self._continuity = ContinuityReward()
        self._length = LengthReward()

    def __call__(self, completions: list, **kwargs: Any) -> List[float]:
        r_out = self._outcome(completions, **kwargs)
        r_top = self._topo(completions, **kwargs)
        r_con = self._continuity(completions, **kwargs)
        r_len = self._length(completions, **kwargs)
        w = self.WEIGHTS
        return [
            round(w["outcome"] * a + w["topo"] * b + w["continuity"] * c + w["length"] * d, 4)
            for a, b, c, d in zip(r_out, r_top, r_con, r_len)
        ]


class TopoOnlyReward(ORM):
    """Ablation: topology reward only (weight=1.0)."""

    def __init__(self, **kwargs) -> None:
        self._topo = TopoReward()

    def __call__(self, completions: list, **kwargs: Any) -> List[float]:
        return self._topo(completions, **kwargs)


class OutcomeLengthReward(ORM):
    """Length-aware GRPO baseline: outcome + format + explicit length control,
    but no topology and no continuity signal.

    This is the rebuttal baseline that isolates whether TopoPRM's gains come
    from topology-aware supervision or merely from brevity pressure
    (HxUk W2, B5w7 W3). It applies the same length regularizer as the full
    TopoPRM composite reward, so any TopoPRM advantage over this row is
    attributable to structure, not length.
    """

    WEIGHTS = {"outcome": 0.70, "format": 0.15, "length": 0.15}

    def __init__(self, **kwargs) -> None:
        self._outcome = OutcomeReward()
        self._format = FormatReward()
        self._length = LengthReward()

    def __call__(self, completions: list, **kwargs: Any) -> List[float]:
        r_out = self._outcome(completions, **kwargs)
        r_fmt = self._format(completions, **kwargs)
        r_len = self._length(completions, **kwargs)
        w = self.WEIGHTS
        return [
            round(w["outcome"] * a + w["format"] * b + w["length"] * c, 4)
            for a, b, c in zip(r_out, r_fmt, r_len)
        ]


orms["ablation_outcome_only"] = OutcomeOnlyReward
orms["ablation_no_topo"] = NoTopoReward
orms["ablation_no_continuity"] = NoContinuityReward
orms["ablation_no_format"] = NoFormatReward
orms["ablation_topo_only"] = TopoOnlyReward
orms["ablation_outcome_length"] = OutcomeLengthReward


# Alias for the verifier-style topology reward after R_topo refactor
orms["ablation_verifiable_topo"] = TopoOnlyReward
