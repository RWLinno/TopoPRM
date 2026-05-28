"""TopoPRM composite rewards for ms-swift GRPO.

This module implements the three reward aggregators referenced in the paper:

* :class:`TopoCompositeReward` -- linear mix of outcome/format/topo/continuity/length.
* :class:`TopoHierarchicalReward` -- multiplicative form ``r_base * (1 + alpha*r_topo + (1-alpha)*r_cont)``.
  This is the main reward used for the 9B/7B TopoPRM models reported in the paper.
* :class:`TopoGatedReward` -- lexicographic "outcome first, process tie-break" form.

Principal hyperparameters (the only knobs appearing in the paper equations):

* ``alpha`` (``TOPO_HIER_ALPHA``): topology vs continuity mix weight (Eq. R_hier).
* ``BASE_WEIGHTS`` class constants (0.70/0.15/0.15 for outcome/format/length).
* ``TopoReward`` internal weights for the five DAG indicators
  (valid / acyclic / no_orphan / direction / step_align / ref_edge_f1).
* Orphan-support edge weights (virtual=1.0, double_barrier=0.5, solid=0.3) in
  :func:`src.reward.topo_reward.TopoReward._orphan_conclusion_ratio`.
* ``TOPO_HIER_BASE_FLOOR`` (0.05): a bug-fix floor documented in the paper
  appendix; prevents zero-variance rollout groups when the outcome reward is 0.

Everything else (``TOPO_*_NOISE_EPS``, ``TOPO_*_MIN_STD``, ``TOPO_HIER_REWARD_TEMP``,
``TOPO_DYNAMIC_*``) is ablation-only: defaults are set so that these are no-ops,
and the main results in the paper are reproduced without any of them.  GRPO
already performs group-wise advantage normalization via ms-swift
(``scale_rewards='group'`` by default); duplicating that with explicit std-floor
noise injection was deprecated in the 2026-04-23 cleanup pass.
"""

from __future__ import annotations

import math
import os
from typing import Any, Iterable, Optional

from swift.rewards import ORM, orms

from src.reward.continuity_reward import ContinuityReward
from src.reward.format_reward import FormatReward
from src.reward.outcome_reward import OutcomeReward
from src.reward.reward_config import RewardConfig, env_float, env_int
from src.reward.topo_reward import TopoReward
from src.reward.utils import completion_to_text


def _load_ablation_config(path: Optional[str] = None) -> dict:
    """Load ablation config from YAML file.

    Checks (in order): explicit path, TOPO_ABLATION_CONFIG env var, default path.
    Returns empty dict if no config found.
    """
    try:
        import yaml
    except ImportError:
        return {}

    candidates = [
        path,
        os.environ.get("TOPO_ABLATION_CONFIG"),
    ]
    for p in candidates:
        if p and os.path.exists(p):
            with open(p) as f:
                data = yaml.safe_load(f) or {}
            return data.get("ablation", data)
    return {}


class LengthReward(ORM):
    """Length-penalty reward.

    Default unit is characters (preserves the v1 released behaviour):
        * len(text) <= LOW          -> 1.0
        * LOW < len(text) < HIGH    -> linear decay from 1.0 to 0.0
        * len(text) >= HIGH         -> 0.0

    Set the env var ``TOPO_LENGTH_UNIT=tokens`` to switch to whitespace-
    delimited token counts; in that case ``TOPO_LENGTH_LOW`` and
    ``TOPO_LENGTH_HIGH`` are interpreted as token thresholds.  This is a
    P5 patch (see docs/method_diagnosis_2026-05-14.md): char-based
    thresholds saturate at 0.0 on long-CoT traces (AIME ~6k chars/trace),
    which silently drops the length signal from the hierarchical reward.
    """

    LOW: int = RewardConfig.LENGTH_LOW
    HIGH: int = RewardConfig.LENGTH_HIGH
    UNIT: str = RewardConfig.LENGTH_UNIT

    def __call__(
        self,
        completions: list,
        **kwargs: Any,
    ) -> list[float]:
        rewards: list[float] = []
        for completion in completions:
            text = completion_to_text(completion)
            if self.UNIT == "tokens":
                length = len(text.split())
            else:
                length = len(text)
            if length <= self.LOW:
                rewards.append(1.0)
            elif length >= self.HIGH:
                rewards.append(0.0)
            else:
                rewards.append(1.0 - (length - self.LOW) / (self.HIGH - self.LOW))
        return rewards


class _ZeroReward(ORM):
    """Dummy reward that always returns 0.0 — used to disable components."""

    def __call__(self, completions: list, **kwargs: Any) -> list[float]:
        return [0.0] * len(completions)


class _SafeCompositeBase(ORM):
    """Shared utility for robust composite rewards."""

    LOG_EVERY: int = RewardConfig.TOPO_REWARD_LOG_EVERY
    DYNAMIC_REWARD: bool = RewardConfig.TOPO_DYNAMIC_REWARD
    DYNAMIC_ETA: float = RewardConfig.TOPO_DYNAMIC_ETA
    MIN_WEIGHT: float = RewardConfig.TOPO_DYNAMIC_MIN_WEIGHT
    OUTCOME_FLOOR: float = RewardConfig.TOPO_DYNAMIC_OUTCOME_FLOOR

    def __init__(self, ablation_config: Optional[str] = None, **kwargs: Any) -> None:
        super().__init__()
        self._ablation = _load_ablation_config(ablation_config)
        self._outcome = OutcomeReward()
        self._format = FormatReward()
        self._continuity = ContinuityReward()
        self._length = LengthReward()
        self._num_calls = 0

        # Conditionally disable components based on ablation config
        use_topo = self._ablation.get("use_topo_reward", True)
        topo_scorer = self._ablation.get("topo_scorer", "rule_based")

        if not use_topo:
            self._topo = _ZeroReward()
        elif topo_scorer in ("gat", "hybrid"):
            try:
                from src.reward.gat_topo_reward import GATTopoReward
                os.environ.setdefault("TOPO_SCORER", topo_scorer)
                self._topo = GATTopoReward()
            except ImportError:
                self._topo = TopoReward()
        else:
            self._topo = TopoReward()

        if not self._ablation.get("use_continuity_reward", True):
            self._continuity = _ZeroReward()
        if not self._ablation.get("use_format_reward", True):
            self._format = _ZeroReward()
        if not self._ablation.get("use_length_reward", True):
            self._length = _ZeroReward()

    @staticmethod
    def _sanitize(x: Any, default: float = 0.0) -> float:
        try:
            v = float(x)
        except Exception:
            return default
        if math.isnan(v) or math.isinf(v):
            return default
        return v

    @staticmethod
    def _clip01(x: float) -> float:
        return max(0.0, min(1.0, x))

    @staticmethod
    def _mean(xs: list[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    @staticmethod
    def _std(xs: list[float]) -> float:
        if not xs:
            return 0.0
        mu = sum(xs) / len(xs)
        var = sum((x - mu) ** 2 for x in xs) / len(xs)
        return math.sqrt(max(var, 0.0))

    @classmethod
    def _inject_std_floor(
        cls,
        rewards: list[float],
        min_std: float,
        noise_eps: float,
    ) -> list[float]:
        """If batch reward std < min_std, inject zero-mean Gaussian noise with
        scale ``noise_eps`` so that GRPO advantage estimation has non-zero
        variance to learn from. Disabled when ``noise_eps <= 0``.
        """
        if not rewards or noise_eps <= 0:
            return rewards
        std_val = cls._std(rewards)
        if std_val >= min_std:
            return rewards
        import random

        return [r + random.gauss(0.0, noise_eps) for r in rewards]

    def _corr(self, xs: list[float], ys: list[float]) -> float:
        if not xs or not ys or len(xs) != len(ys):
            return 0.0
        mx = self._mean(xs)
        my = self._mean(ys)
        vx = sum((x - mx) ** 2 for x in xs)
        vy = sum((y - my) ** 2 for y in ys)
        if vx <= 1e-12 or vy <= 1e-12:
            return 0.0
        cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        return cov / math.sqrt(vx * vy)

    def _renorm_weights(self, weights: dict[str, float]) -> dict[str, float]:
        if not weights:
            return {}
        keys = list(weights.keys())
        vals = {k: max(self.MIN_WEIGHT, self._sanitize(weights.get(k, 0.0), default=0.0)) for k in keys}
        s = sum(vals.values())
        if s <= 1e-12:
            return {k: 1.0 / len(keys) for k in keys}
        vals = {k: vals[k] / s for k in keys}

        if "outcome" in vals and vals["outcome"] < self.OUTCOME_FLOOR:
            deficit = self.OUTCOME_FLOOR - vals["outcome"]
            others = [k for k in keys if k != "outcome"]
            others_sum = sum(vals[k] for k in others)
            if others and others_sum > 1e-12:
                for k in others:
                    vals[k] = max(0.0, vals[k] - deficit * (vals[k] / others_sum))
                vals["outcome"] = self.OUTCOME_FLOOR
                s2 = sum(vals.values())
                if s2 > 1e-12:
                    vals = {k: vals[k] / s2 for k in keys}
        return vals

    def _resolve_dynamic_weights(
        self,
        base_weights: dict[str, float],
        outcome_scores: list[float],
        components: dict[str, list[float]],
    ) -> dict[str, float]:
        base = self._renorm_weights(base_weights)
        if not self.DYNAMIC_REWARD:
            return base

        reliability: dict[str, float] = {}
        for name in base:
            if name == "outcome":
                reliability[name] = 1.0
                continue
            xs = components.get(name, [])
            corr = max(0.0, self._corr(xs, outcome_scores))
            spread = min(1.0, self._std(xs) * 2.0)
            reliability[name] = 0.5 * corr + 0.5 * spread

        raw = {k: base[k] * (1.0 + reliability.get(k, 0.0)) for k in base}
        adapted = self._renorm_weights(raw)
        eta = self._clip01(self.DYNAMIC_ETA)
        blended = {k: (1.0 - eta) * base[k] + eta * adapted[k] for k in base}
        return self._renorm_weights(blended)

    def _components(
        self,
        completions: list,
        solution: Any = None,
        reference_dag: Any = None,
        **kwargs: Any,
    ) -> tuple[list[float], list[float], list[float], list[float], list[float]]:
        outcome_scores = [self._clip01(self._sanitize(v)) for v in self._outcome(completions, solution=solution, **kwargs)]
        format_scores = [self._clip01(self._sanitize(v)) for v in self._format(completions, **kwargs)]
        topo_scores = [self._clip01(self._sanitize(v)) for v in self._topo(completions, reference_dag=reference_dag, **kwargs)]
        continuity_scores = [self._clip01(self._sanitize(v)) for v in self._continuity(completions, **kwargs)]
        length_scores = [self._clip01(self._sanitize(v)) for v in self._length(completions, **kwargs)]
        return outcome_scores, format_scores, topo_scores, continuity_scores, length_scores

    def _maybe_log_stats(self, rewards: Iterable[float], tag: str) -> None:
        if not self.LOG_EVERY:
            return
        self._num_calls += 1
        if self._num_calls % self.LOG_EVERY:
            return
        vals = [self._sanitize(v) for v in rewards]
        if not vals:
            return
        mean_v = sum(vals) / len(vals)
        min_v = min(vals)
        max_v = max(vals)
        std_v = self._std(vals)
        print(f"[{tag}] n={len(vals)} mean={mean_v:.4f} min={min_v:.4f} max={max_v:.4f} std={std_v:.6f}")
        # Collapse warning
        if std_v < 0.005 and len(vals) > 2:
            print(f"[{tag}] WARNING: reward collapse detected (std={std_v:.6f} < 0.005). "
                  f"Consider using topo_hierarchical with anti-collapse or increasing temperature.")

    def _maybe_log_components(
        self,
        tag: str,
        outcome_scores: list[float],
        format_scores: list[float],
        topo_scores: list[float],
        continuity_scores: list[float],
        length_scores: list[float],
        rewards: list[float],
        weights: dict[str, float] | None = None,
    ) -> None:
        """Print per-component reward means every LOG_EVERY calls."""
        if not self.LOG_EVERY:
            return
        if not outcome_scores:
            return

        self._num_calls += 1
        if self._num_calls % self.LOG_EVERY:
            return

        w_msg = ""
        if weights:
            ordered = ["outcome", "format", "topo", "continuity", "length"]
            w_msg = " " + " ".join(
                f"w_{k}={weights.get(k, 0.0):.3f}" for k in ordered if k in weights
            )

        print(
            f"[{tag}] "
            f"outcome={self._mean(outcome_scores):.4f} "
            f"format={self._mean(format_scores):.4f} "
            f"topo={self._mean(topo_scores):.4f} "
            f"continuity={self._mean(continuity_scores):.4f} "
            f"length={self._mean(length_scores):.4f} "
            f"total={self._mean(rewards):.4f} "
            f"n={len(rewards)}{w_msg}"
        )


class TopoCompositeReward(_SafeCompositeBase):
    """Composite aggregator for deterministic PRM signals.

    This module mixes outcome/process/auxiliary rewards while preserving
    outcome primacy through dynamic weighting constraints. It is the default
    plugin-friendly implementation used by the current GRPO pipeline.
    """

    WEIGHTS: dict[str, float] = {
        "outcome": 0.40,
        "format": 0.15,
        "topo": 0.20,
        "continuity": 0.15,
        "length": 0.10,
    }

    def __call__(
        self,
        completions: list,
        solution: Any = None,
        reference_dag: Any = None,
        **kwargs: Any,
    ) -> list[float]:
        outcome_scores, format_scores, topo_scores, continuity_scores, length_scores = self._components(
            completions,
            solution=solution,
            reference_dag=reference_dag,
            **kwargs,
        )

        w = self._resolve_dynamic_weights(
            self.WEIGHTS,
            outcome_scores,
            {
                "format": format_scores,
                "topo": topo_scores,
                "continuity": continuity_scores,
                "length": length_scores,
            },
        )
        rewards: list[float] = []
        for o, f, t, c, l in zip(outcome_scores, format_scores, topo_scores, continuity_scores, length_scores):
            r = (
                w["outcome"] * o
                + w["format"] * f
                + w["topo"] * t
                + w["continuity"] * c
                + w["length"] * l
            )
            rewards.append(round(r, 6))
        # Ablation-only std-floor (default NOISE_EPS=0 -> no-op).  See module docstring.
        rewards = self._inject_std_floor(
            rewards,
            RewardConfig.TOPO_COMPOSITE_MIN_STD,
            RewardConfig.TOPO_COMPOSITE_NOISE_EPS,
        )
        rewards = [round(self._clip01(r), 6) for r in rewards]
        self._maybe_log_components(
            "topo_composite_linear/components",
            outcome_scores,
            format_scores,
            topo_scores,
            continuity_scores,
            length_scores,
            rewards,
            weights=w,
        )
        self._maybe_log_stats(rewards, "topo_composite_linear")
        return rewards


class TopoHierarchicalReward(_SafeCompositeBase):
    """Hierarchical reward aggregation with batch rescaling and anti-collapse.

    Core formula:
        R = R_base * (1 + alpha * scale(R_topo) + (1-alpha) * scale(R_cont))

    where R_base = w_o * outcome + w_f * format + w_l * length,
    and scale() is batch-level min-max rescaling to [0, 1].

    Anti-collapse mechanisms:
    - Batch-level min-max rescaling ensures reward components have spread
    - Minimum reward std guard injects small noise when variance collapses
    - Reward temperature rescaling amplifies small differences
    """

    BASE_WEIGHTS: dict[str, float] = {
        "outcome": 0.70,
        "format": 0.15,
        "length": 0.15,
    }
    # Defaults; overridden by YAML ablation config or env vars
    ALPHA: float = RewardConfig.TOPO_HIER_ALPHA
    NOISE_EPS: float = RewardConfig.TOPO_HIER_NOISE_EPS
    MIN_STD: float = RewardConfig.TOPO_HIER_MIN_STD
    REWARD_TEMP: float = RewardConfig.TOPO_HIER_REWARD_TEMP
    BASE_FLOOR: float = RewardConfig.TOPO_HIER_BASE_FLOOR

    def __init__(self, ablation_config: Optional[str] = None, **kwargs: Any) -> None:
        super().__init__(ablation_config=ablation_config, **kwargs)
        cfg = self._ablation
        self.ALPHA = float(os.environ.get("TOPO_HIER_ALPHA", cfg.get("alpha", self.ALPHA)))
        self.NOISE_EPS = float(os.environ.get("TOPO_HIER_NOISE_EPS", cfg.get("reward_noise_eps", self.NOISE_EPS)))
        self.MIN_STD = float(os.environ.get("TOPO_HIER_MIN_STD", cfg.get("min_reward_std", self.MIN_STD)))
        self.REWARD_TEMP = float(os.environ.get("TOPO_HIER_REWARD_TEMP", cfg.get("reward_temperature", self.REWARD_TEMP)))
        self.BASE_FLOOR = float(os.environ.get("TOPO_HIER_BASE_FLOOR", cfg.get("base_floor", self.BASE_FLOOR)))

    @staticmethod
    def _batch_rescale(scores: list[float]) -> list[float]:
        """Min-max rescale a batch of scores to [0, 1] with epsilon guard.

        When ``TOPO_RESCALE_PATCH`` is enabled (default off), group with
        true spread below ``TOPO_RESCALE_MIN_SPAN`` (default 0.05) are
        returned as a constant 0.5 vector instead of being stretched to
        the full [0, 1] range.  This prevents topology/continuity noise
        from dominating advantages on outcome-saturated batches
        (easy GSM8K / MATH500 rollouts); see docs/method_diagnosis.md §R2.
        """
        if not scores:
            return []
        lo = min(scores)
        hi = max(scores)
        span = hi - lo
        if RewardConfig.TOPO_RESCALE_PATCH:
            if span < RewardConfig.TOPO_RESCALE_MIN_SPAN:
                return [0.5] * len(scores)
        else:
            if span < 1e-8:
                return [0.5] * len(scores)
        return [(s - lo) / span for s in scores]

    @staticmethod
    def _reward_temperature_scale(scores: list[float], temperature: float) -> list[float]:
        """Apply temperature scaling to amplify reward differences.

        Centres scores around their mean, scales by temperature, then shifts back.
        """
        if not scores or temperature <= 0:
            return scores
        mu = sum(scores) / len(scores)
        return [mu + (s - mu) * temperature for s in scores]

    def __call__(
        self,
        completions: list,
        solution: Any = None,
        reference_dag: Any = None,
        **kwargs: Any,
    ) -> list[float]:
        outcome_scores, format_scores, topo_scores, continuity_scores, length_scores = self._components(
            completions,
            solution=solution,
            reference_dag=reference_dag,
            **kwargs,
        )

        bw = self.BASE_WEIGHTS
        alpha = self._clip01(self.ALPHA)

        # Batch-level rescaling for process rewards
        topo_scaled = self._batch_rescale(topo_scores)
        cont_scaled = self._batch_rescale(continuity_scores)

        floor = max(0.0, float(self.BASE_FLOOR))
        # P2: switch from additive base (w_o*o + w_f*f + w_l*l) to a truly
        # multiplicative aggregation where r_base = 0 iff outcome = 0.
        # Default 'additive' preserves released-checkpoint behaviour.
        agg_mode = RewardConfig.TOPO_HIER_AGG.lower()
        if agg_mode not in ("additive", "multiplicative"):
            agg_mode = "additive"
        rewards: list[float] = []
        for o, f, t, c, l in zip(outcome_scores, format_scores, topo_scaled, cont_scaled, length_scores):
            if agg_mode == "multiplicative":
                # Hard correctness primacy: outcome=0 -> r_base=0, so the
                # topology gain cannot lift answer-incorrect traces above
                # answer-correct ones.  Format/length become gate factors
                # in [f_min, 1], where f_min keeps small signal for the
                # advantage normaliser.
                f_gate = 0.5 + 0.5 * f
                l_gate = 0.5 + 0.5 * l
                r_base = o * f_gate * l_gate
                r_base_floored = r_base  # no floor in multiplicative mode
            else:
                r_base = bw["outcome"] * o + bw["format"] * f + bw["length"] * l
                # Floor on r_base so that topology gain is never multiplied by zero
                # when outcome=format=length=0 (this was responsible for ~36% of
                # zero-variance rollout groups in v3b — see method_diagnosis_2026-04-22.md).
                r_base_floored = max(r_base, floor)
            gain = 1.0 + alpha * t + (1.0 - alpha) * c
            r = r_base_floored * gain
            rewards.append(round(r, 6))

        # Temperature rescaling to amplify differences
        if self.REWARD_TEMP != 1.0:
            rewards = self._reward_temperature_scale(rewards, self.REWARD_TEMP)

        # Ablation-only: inject Gaussian noise when batch std < MIN_STD.
        # Default NOISE_EPS=0 makes this a no-op in main runs (see module
        # docstring).  GRPO's scale_rewards='group' already standardizes
        # advantages; this hook is only retained for reproducing the
        # 2026-04-20 reward-collapse diagnostic experiments.
        rewards = self._inject_std_floor(rewards, self.MIN_STD, self.NOISE_EPS)

        # Final clip
        rewards = [round(self._clip01(r), 6) for r in rewards]

        self._maybe_log_components(
            "topo_hierarchical/components",
            outcome_scores,
            format_scores,
            topo_scores,
            continuity_scores,
            length_scores,
            rewards,
        )
        self._maybe_log_stats(rewards, "topo_hierarchical")
        return rewards


class TopoMultiplicativeGateReward(_SafeCompositeBase):
    """Hierarchical multiplicative gate for multi-source reward aggregation.

    r = r_base * (1 + alpha * r_topo + beta * r_cont)

    where r_base combines outcome/format/length. This enforces answer-centric
    optimization while letting topology and continuity act as controlled gain.
    """

    BASE_WEIGHTS: dict[str, float] = {
        "outcome": 0.70,
        "format": 0.15,
        "length": 0.15,
    }
    ALPHA: float = 0.30
    BETA: float = 0.25

    def __call__(
        self,
        completions: list,
        solution: Any = None,
        reference_dag: Any = None,
        **kwargs: Any,
    ) -> list[float]:
        outcome_scores, format_scores, topo_scores, continuity_scores, length_scores = self._components(
            completions,
            solution=solution,
            reference_dag=reference_dag,
            **kwargs,
        )

        bw = self.BASE_WEIGHTS
        rewards: list[float] = []
        for o, f, t, c, l in zip(outcome_scores, format_scores, topo_scores, continuity_scores, length_scores):
            r_base = bw["outcome"] * o + bw["format"] * f + bw["length"] * l
            gain = 1.0 + self.ALPHA * t + self.BETA * c
            r = self._clip01(r_base * gain)
            rewards.append(round(r, 6))
        self._maybe_log_components(
            "topo_composite_mulgate/components",
            outcome_scores,
            format_scores,
            topo_scores,
            continuity_scores,
            length_scores,
            rewards,
        )
        self._maybe_log_stats(rewards, "topo_composite_mulgate")
        return rewards


class TopoConfidenceGateReward(_SafeCompositeBase):
    """Outcome-confidence gated process rewards.

    If outcome correctness is low, process rewards receive weaker gain to reduce
    reward hacking from structurally plausible but answer-incorrect traces.
    """

    BASE_WEIGHTS: dict[str, float] = {
        "outcome": 0.60,
        "format": 0.20,
        "length": 0.20,
    }
    HIGH_GAIN: float = 0.60
    LOW_GAIN: float = 0.20
    CORRECT_THRESHOLD: float = 0.66

    def __call__(
        self,
        completions: list,
        solution: Any = None,
        reference_dag: Any = None,
        **kwargs: Any,
    ) -> list[float]:
        outcome_scores, format_scores, topo_scores, continuity_scores, length_scores = self._components(
            completions,
            solution=solution,
            reference_dag=reference_dag,
            **kwargs,
        )

        bw = self.BASE_WEIGHTS
        rewards: list[float] = []
        for o, f, t, c, l in zip(outcome_scores, format_scores, topo_scores, continuity_scores, length_scores):
            r_base = bw["outcome"] * o + bw["format"] * f + bw["length"] * l
            process = 0.5 * t + 0.5 * c
            gain = self.HIGH_GAIN if o >= self.CORRECT_THRESHOLD else self.LOW_GAIN
            r = self._clip01(r_base + gain * process)
            rewards.append(round(r, 6))
        self._maybe_log_components(
            "topo_composite_confgate/components",
            outcome_scores,
            format_scores,
            topo_scores,
            continuity_scores,
            length_scores,
            rewards,
        )
        self._maybe_log_stats(rewards, "topo_composite_confgate")
        return rewards


class TopoClippedScalarReward(_SafeCompositeBase):
    """Constrained scalarization with per-component clipping.

    This is a robust linear baseline that clips each component into predefined
    trust ranges before weighted aggregation to avoid single-term domination.
    """

    WEIGHTS: dict[str, float] = {
        "outcome": 0.45,
        "format": 0.15,
        "topo": 0.20,
        "continuity": 0.15,
        "length": 0.05,
    }

    CLIP_MIN: dict[str, float] = {
        "outcome": 0.0,
        "format": 0.0,
        "topo": 0.10,
        "continuity": 0.10,
        "length": 0.0,
    }

    CLIP_MAX: dict[str, float] = {
        "outcome": 1.0,
        "format": 1.0,
        "topo": 0.95,
        "continuity": 0.95,
        "length": 1.0,
    }

    def _clip_component(self, name: str, value: float) -> float:
        lo = self.CLIP_MIN[name]
        hi = self.CLIP_MAX[name]
        return max(lo, min(hi, value))

    def __call__(
        self,
        completions: list,
        solution: Any = None,
        reference_dag: Any = None,
        **kwargs: Any,
    ) -> list[float]:
        outcome_scores, format_scores, topo_scores, continuity_scores, length_scores = self._components(
            completions,
            solution=solution,
            reference_dag=reference_dag,
            **kwargs,
        )

        w = self.WEIGHTS
        rewards: list[float] = []
        for o, f, t, c, l in zip(outcome_scores, format_scores, topo_scores, continuity_scores, length_scores):
            o = self._clip_component("outcome", o)
            f = self._clip_component("format", f)
            t = self._clip_component("topo", t)
            c = self._clip_component("continuity", c)
            l = self._clip_component("length", l)
            r = w["outcome"] * o + w["format"] * f + w["topo"] * t + w["continuity"] * c + w["length"] * l
            rewards.append(round(self._clip01(r), 6))
        self._maybe_log_components(
            "topo_composite_clipped/components",
            outcome_scores,
            format_scores,
            topo_scores,
            continuity_scores,
            length_scores,
            rewards,
        )
        self._maybe_log_stats(rewards, "topo_composite_clipped")
        return rewards



class TopoSCAEReward(_SafeCompositeBase):
    """Correctness-first stratified shaping (SCAE-style) for GRPO plugins.

    Note: ms-swift does not expose GRPO internal advantage computation in this
    repository. This class implements stratified clipping at reward-output level
    as a practical drop-in approximation for correctness-first reward shaping.
    In the paper framing, this is a supporting optimization mechanism rather
    than a headline contribution.
    """

    EPS: float = 1e-6
    OUTCOME_THRESHOLD: float = 0.66
    POS_CLIP: tuple[float, float] = (0.0, 1.5)
    NEG_CLIP: tuple[float, float] = (-1.5, 0.0)

    def __init__(self, base_reward: ORM | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._base = base_reward or TopoCompositeReward()

    @staticmethod
    def _normalize(vals: list[float]) -> list[float]:
        if not vals:
            return []
        mu = sum(vals) / len(vals)
        var = sum((v - mu) ** 2 for v in vals) / len(vals)
        std = math.sqrt(var)
        return [(v - mu) / (std + TopoSCAEReward.EPS) for v in vals]

    @staticmethod
    def _clip(v: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, v))

    def __call__(
        self,
        completions: list,
        solution: Any = None,
        reference_dag: Any = None,
        **kwargs: Any,
    ) -> list[float]:
        base_rewards = [self._sanitize(v) for v in self._base(completions, solution=solution, reference_dag=reference_dag, **kwargs)]
        outcome_scores = [self._clip01(self._sanitize(v)) for v in self._outcome(completions, solution=solution, **kwargs)]

        pos_idx = [i for i, o in enumerate(outcome_scores) if o >= self.OUTCOME_THRESHOLD]
        neg_idx = [i for i, o in enumerate(outcome_scores) if o < self.OUTCOME_THRESHOLD]

        pos_vals = [base_rewards[i] for i in pos_idx]
        neg_vals = [base_rewards[i] for i in neg_idx]
        pos_norm = self._normalize(pos_vals)
        neg_norm = self._normalize(neg_vals)

        shaped = [0.0] * len(base_rewards)

        if RewardConfig.SCAE_PRESERVE_OUTCOME:
            # P1: preserve outcome magnitude across strata.
            # B+ shaped values live in [floor_pos, POS_CLIP[1]]
            # B- shaped values live in [NEG_CLIP[0], -floor_neg]
            # This guarantees min(B+) > max(B-) when floor_pos > 0 and
            # floor_neg > 0, so the cross-stratum ordering is never violated.
            floor_pos = RewardConfig.SCAE_FLOOR_POS
            floor_neg = RewardConfig.SCAE_FLOOR_NEG
            clip_hi = self.POS_CLIP[1]
            clip_lo = self.NEG_CLIP[0]
            for k, i in enumerate(pos_idx):
                # Map normalized value from [0, POS_CLIP[1]] to [floor_pos, clip_hi]
                raw = self._clip(pos_norm[k], self.POS_CLIP[0], clip_hi)
                shaped[i] = round(floor_pos + (clip_hi - floor_pos) * (raw / clip_hi) if clip_hi > 0 else floor_pos, 6)
            for k, i in enumerate(neg_idx):
                # Map normalized value from [NEG_CLIP[0], 0] to [clip_lo, -floor_neg]
                # raw=-1.5 → clip_lo=-1.5; raw=0 → -floor_neg=-0.3
                raw = self._clip(neg_norm[k], clip_lo, self.NEG_CLIP[1])
                if clip_lo < 0:
                    t = raw / clip_lo  # t in [0, 1], t=1 when raw=clip_lo
                    shaped[i] = round(-floor_neg + (clip_lo + floor_neg) * t, 6)
                else:
                    shaped[i] = round(-floor_neg, 6)
        else:
            # Default (v1): raw per-stratum normalized + clipped values.
            for k, i in enumerate(pos_idx):
                shaped[i] = round(self._clip(pos_norm[k], self.POS_CLIP[0], self.POS_CLIP[1]), 6)
            for k, i in enumerate(neg_idx):
                shaped[i] = round(self._clip(neg_norm[k], self.NEG_CLIP[0], self.NEG_CLIP[1]), 6)

        self._maybe_log_stats(shaped, "topo_composite_scae")
        return shaped



class CompositeRewardAggregator(TopoCompositeReward):
    """Alias with paper-aligned naming."""


class CorrectnessFirstShapingReward(TopoSCAEReward):
    """Alias with paper-aligned naming."""


def get_reward_func(reward_type: str = "composite") -> ORM:
    """Return a reward class instance by name."""
    registry: dict[str, type[ORM]] = {
        "composite": TopoCompositeReward,
        "composite_aggregator": CompositeRewardAggregator,
        "outcome": OutcomeReward,
        "format": FormatReward,
        "topo": TopoReward,
        "continuity": ContinuityReward,
        "length": LengthReward,
        "mulgate": TopoMultiplicativeGateReward,
        "confgate": TopoConfidenceGateReward,
        "clipped": TopoClippedScalarReward,
        "scae": TopoSCAEReward,
        "correctness_first": CorrectnessFirstShapingReward,
        "deterministic_prm_composite": CompositeRewardAggregator,
        "deterministic_prm_correctness_first": CorrectnessFirstShapingReward,
    }
    if reward_type not in registry:
        raise ValueError(f"Unknown reward_type {reward_type!r}. Choose from {sorted(registry.keys())}.")
    return registry[reward_type]()


class TopoGatedReward(_SafeCompositeBase):
    r"""Outcome-gated process reward with principled lexicographic design.

    Motivation (cross-scale failure analysis, 2026-04-06):
      GRPO only cares about *relative ordering within a batch*.  The reward
      must satisfy:

      P1  **Outcome primacy** — a generation with higher outcome always
          ranks above one with lower outcome, regardless of process quality.
      P2  **Process as tiebreaker** — among generations with equal outcome,
          better reasoning structure ranks higher.
      P3  **Curriculum signal** — when outcome = 0 for all generations
          (weak model regime), format compliance provides a warm-start
          gradient so the model can learn to produce parseable outputs first.

    Design (lexicographic reward):

    .. math::
        R = \text{outcome} + \delta \cdot \text{format} \cdot (1 + \varepsilon \cdot q)

    where:
      * ``outcome`` ∈ {0, 0.333, 0.5, 0.667, 1.0} — answer correctness
      * ``format``  ∈ {0, 0.3, 1.0} — structural compliance
      * ``q`` ∈ [0, 1] — batch-normalised process quality (mean of
        rescaled topology and continuity scores)
      * δ = 0.1 — **derived, not tuned**: the minimum gap between
        adjacent outcome values is 0.167 (= 0.667 − 0.5); setting
        δ < 0.167 guarantees that the format+process term can never
        reverse the outcome ranking (see proof below).
      * ε = 0.5 — influence bound for process within the format tier.

    Ranking-preservation proof:
      Max contribution of the second term = δ × 1.0 × (1 + ε × 1) = 0.15.
      Min gap between distinct outcome values = 0.167.
      Since 0.15 < 0.167, outcome ordering is strictly preserved.  ∎

    Truncation handling (FM1):
      When a generation hits ``max_completion_length``, its DAG is
      incomplete.  We set q = 0.5 (batch-neutral) to avoid rewarding
      or penalising based on a garbage topology signal.

    Variance guarantee (FM2):
      Batch-level min-max rescaling of topo and continuity ensures
      that q always has spread, even when raw scores are near-constant.
    """

    # δ: derived from OutcomeReward's discrete gap structure.
    # OutcomeReward values ∈ {0, 1/3, 1/2, 2/3, 1}; min adjacent gap = 1/6 ≈ 0.167.
    # Any δ < 1/6 preserves outcome ranking.  We use 0.1 (< 0.167).
    DELTA: float = env_float("TOPO_GATED_DELTA", 0.1)

    # ε: influence bound for process quality within the format tier.
    # Must satisfy δ × (1 + ε) < min_outcome_gap = 0.167.
    # 0.1 × 1.5 = 0.15 < 0.167 ✓
    EPSILON: float = env_float("TOPO_GATED_EPSILON", 0.5)

    # Topo vs continuity mix: 0.0 = continuity-only, 0.5 = equal, 1.0 = topo-only.
    # Set TOPO_GATED_TOPO_W=0 for ablation without topology signal.
    TOPO_WEIGHT: float = env_float("TOPO_GATED_TOPO_W", 0.5)

    @staticmethod
    def _batch_rescale(scores: list[float]) -> list[float]:
        """Min-max rescale to [0, 1]; constant batches map to 0.5."""
        if not scores:
            return []
        lo, hi = min(scores), max(scores)
        span = hi - lo
        if span < 1e-8:
            return [0.5] * len(scores)
        return [(s - lo) / span for s in scores]

    @staticmethod
    def _is_truncated(completion) -> bool:
        """A generation is truncated iff it lacks a closing </answer> tag.

        This is model-agnostic and independent of max_completion_length:
        a well-formed output always ends with </answer>.  If the tag is
        missing, the generation was cut short before the model finished.
        """
        text = completion_to_text(completion)
        return "</answer>" not in text

    def __call__(
        self,
        completions: list,
        solution: Any = None,
        reference_dag: Any = None,
        **kwargs: Any,
    ) -> list[float]:
        outcome_scores, fmt_scores, topo_scores, cont_scores, _len = (
            self._components(
                completions,
                solution=solution,
                reference_dag=reference_dag,
                **kwargs,
            )
        )

        n = len(outcome_scores)
        delta = self.DELTA
        eps = self.EPSILON

        # Batch-rescale process signals to guarantee spread
        topo_sc = self._batch_rescale(topo_scores)
        cont_sc = self._batch_rescale(cont_scores)

        # Topo weight: 0.0 = continuity-only, 1.0 = topo-only, 0.5 = equal mix
        topo_w = self.TOPO_WEIGHT

        rewards: list[float] = []
        for i in range(n):
            # Process quality: weighted mix of rescaled topo and continuity
            if self._is_truncated(completions[i]):
                q = 0.5          # neutral for truncated outputs
            else:
                q = topo_w * topo_sc[i] + (1.0 - topo_w) * cont_sc[i]

            # Lexicographic reward:
            #   high-order: outcome (primary signal)
            #   low-order:  δ × format × (1 + ε × q)  (tiebreaker + curriculum)
            r = outcome_scores[i] + delta * fmt_scores[i] * (1.0 + eps * q)
            rewards.append(round(r, 6))

        # Ablation-only std-floor (default NOISE_EPS=0 -> no-op).
        # Kept as a hook for reproducing the gated_qwen35_9b reward_std~=5e-4
        # collapse diagnostic documented in docs/reward_collapse_diagnosis_2026-04-20.md.
        rewards = self._inject_std_floor(
            rewards,
            RewardConfig.TOPO_GATED_MIN_STD,
            RewardConfig.TOPO_GATED_NOISE_EPS,
        )
        rewards = [round(self._clip01(r), 6) for r in rewards]

        self._maybe_log_stats(rewards, "topo_gated")
        return rewards


# ---------------------------------------------------------------------------
# Ablation variants for Table 1
# ---------------------------------------------------------------------------

class OutcomeOnlyReward(TopoHierarchicalReward):
    """Ablation: only outcome + format + length, no topology or continuity."""

    def __call__(self, completions, solution=None, reference_dag=None, **kwargs):
        outcome_scores, format_scores, topo_scores, continuity_scores, length_scores = self._components(
            completions, solution=solution, reference_dag=reference_dag, **kwargs,
        )
        bw = self.BASE_WEIGHTS
        rewards = []
        for o, f, l in zip(outcome_scores, format_scores, length_scores):
            rewards.append(round(bw["outcome"] * o + bw["format"] * f + bw["length"] * l, 6))
        rewards = [round(self._clip01(r), 6) for r in rewards]
        self._maybe_log_stats(rewards, "outcome_only")
        return rewards


class NoTopoReward(TopoHierarchicalReward):
    """Ablation: hierarchical reward but topology component zeroed out."""

    def __call__(self, completions, solution=None, reference_dag=None, **kwargs):
        outcome_scores, format_scores, topo_scores, continuity_scores, length_scores = self._components(
            completions, solution=solution, reference_dag=reference_dag, **kwargs,
        )
        bw = self.BASE_WEIGHTS
        cont_scaled = self._batch_rescale(continuity_scores)
        floor = max(0.0, float(self.BASE_FLOOR))
        rewards = []
        for o, f, c, l in zip(outcome_scores, format_scores, cont_scaled, length_scores):
            r_base = max(bw["outcome"] * o + bw["format"] * f + bw["length"] * l, floor)
            gain = 1.0 + c
            rewards.append(round(r_base * gain, 6))
        rewards = [round(self._clip01(r), 6) for r in rewards]
        self._maybe_log_stats(rewards, "no_topo")
        return rewards


class NoContinuityReward(TopoHierarchicalReward):
    """Ablation: hierarchical reward but continuity component zeroed out."""

    def __call__(self, completions, solution=None, reference_dag=None, **kwargs):
        outcome_scores, format_scores, topo_scores, continuity_scores, length_scores = self._components(
            completions, solution=solution, reference_dag=reference_dag, **kwargs,
        )
        bw = self.BASE_WEIGHTS
        topo_scaled = self._batch_rescale(topo_scores)
        floor = max(0.0, float(self.BASE_FLOOR))
        rewards = []
        for o, f, t, l in zip(outcome_scores, format_scores, topo_scaled, length_scores):
            r_base = max(bw["outcome"] * o + bw["format"] * f + bw["length"] * l, floor)
            gain = 1.0 + t
            rewards.append(round(r_base * gain, 6))
        rewards = [round(self._clip01(r), 6) for r in rewards]
        self._maybe_log_stats(rewards, "no_continuity")
        return rewards


# Register all reward classes in SWIFT's global ``orms`` dict.
orms["topo_gated"] = TopoGatedReward
orms["topo_composite"] = TopoCompositeReward
orms["topo_composite_linear"] = TopoCompositeReward
orms["topo_composite_mulgate"] = TopoMultiplicativeGateReward
orms["topo_composite_confgate"] = TopoConfidenceGateReward
orms["topo_composite_clipped"] = TopoClippedScalarReward
orms["topo_composite_scae"] = TopoSCAEReward
orms["topo_hierarchical"] = TopoHierarchicalReward
orms["deterministic_prm_composite"] = CompositeRewardAggregator
orms["deterministic_prm_correctness_first"] = CorrectnessFirstShapingReward

orms["topo_outcome"] = OutcomeReward
orms["topo_format"] = FormatReward
orms["topo_topo"] = TopoReward
orms["topo_continuity"] = ContinuityReward
orms["topo_length"] = LengthReward

orms["outcome_only"] = OutcomeOnlyReward
orms["no_topo"] = NoTopoReward
orms["no_continuity"] = NoContinuityReward
