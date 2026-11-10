"""Diagnostics for the independence of the two topology reward sources.

TopoPRM treats direction consistency (``q_dir``) and acyclicity (``q_acyc``) as
two *separate* reward coordinates rather than averaging them into a single
topology score.  This module quantifies why that separation matters: on
order-reversed controls the two coordinates move in opposite directions, so a
plain average cancels most of the usable signal.

``source_aggregation_diagnostic`` consumes per-source metrics of the form::

    {source_id: {"original":  {"direction_score": ..., "acyclicity_score": ...},
                 "reversed":  {"direction_score": ..., "acyclicity_score": ...}}}

and reports, with bootstrap confidence intervals:

* ``correlation``                     Pearson r between the two reversed scores.
* ``direction_std`` / ``acyclicity_std``  spread of each coordinate alone.
* ``average_std``                     spread of their mean (the collapsed signal).
* ``average_to_direction_std_ratio``  how much spread survives averaging.
* ``opposite_delta_fraction``         fraction of sources whose two coordinates
  shift in opposite directions relative to the original trace.
"""

from __future__ import annotations

import math
import random
import statistics
from typing import Any

__all__ = ["source_aggregation_diagnostic"]

MetricsBySource = dict[str, dict[str, dict[str, float]]]
_Row = tuple[float, float, float, float]


def _summarize(sample: list[_Row]) -> dict[str, float]:
    reversed_direction = [row[2] for row in sample]
    reversed_acyclicity = [row[3] for row in sample]
    averaged = [
        (direction + acyclicity) / 2
        for direction, acyclicity in zip(reversed_direction, reversed_acyclicity)
    ]
    direction_std = statistics.pstdev(reversed_direction)
    acyclicity_std = statistics.pstdev(reversed_acyclicity)
    average_std = statistics.pstdev(averaged)
    correlation = statistics.correlation(reversed_direction, reversed_acyclicity)
    opposite = sum(
        (
            (reversed_direction_score - original_direction)
            * (reversed_acyclicity_score - original_acyclicity)
        )
        < 0
        for (
            original_direction,
            original_acyclicity,
            reversed_direction_score,
            reversed_acyclicity_score,
        ) in sample
    ) / len(sample)
    return {
        "correlation": correlation,
        "direction_std": direction_std,
        "acyclicity_std": acyclicity_std,
        "average_std": average_std,
        "average_to_direction_std_ratio": (
            average_std / direction_std if direction_std else math.nan
        ),
        "opposite_delta_fraction": opposite,
    }


def source_aggregation_diagnostic(
    metrics_by_source: MetricsBySource,
    reps: int = 1000,
    seed: int = 0,
) -> dict[str, Any]:
    """Point estimates plus bootstrap 95% intervals for source-separation stats.

    Args:
        metrics_by_source: per-source ``original`` / ``reversed`` scores.
        reps: bootstrap resampling repetitions.
        seed: RNG seed, so the diagnostic is reproducible.
    """
    rows: list[_Row] = [
        (
            variants["original"]["direction_score"],
            variants["original"]["acyclicity_score"],
            variants["reversed"]["direction_score"],
            variants["reversed"]["acyclicity_score"],
        )
        for variants in metrics_by_source.values()
    ]
    if not rows:
        raise ValueError("metrics_by_source is empty; nothing to diagnose")

    point = _summarize(rows)
    rng = random.Random(seed)
    draws: dict[str, list[float]] = {name: [] for name in point}
    for _ in range(reps):
        sampled = [rows[rng.randrange(len(rows))] for _ in rows]
        try:
            values = _summarize(sampled)
        except statistics.StatisticsError:
            continue
        for name, value in values.items():
            if math.isfinite(value):
                draws[name].append(value)

    output: dict[str, Any] = {"n": len(rows)}
    for name, value in point.items():
        ordered = sorted(draws[name])
        if not ordered:
            output[name] = {"value": value, "bootstrap_95": [math.nan, math.nan]}
            continue
        output[name] = {
            "value": value,
            "bootstrap_95": [
                ordered[math.floor(0.025 * (len(ordered) - 1))],
                ordered[math.ceil(0.975 * (len(ordered) - 1))],
            ],
        }
    return output
