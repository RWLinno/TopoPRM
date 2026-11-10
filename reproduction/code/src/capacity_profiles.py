"""Canonical candidate capacities for the independent-source Choquet reward."""

from __future__ import annotations

from typing import Final


CHOQUET_CAPACITY_PROFILES: Final = {
    "balanced": {
        "singletons": {
            "outcome": 0.42,
            "format": 0.05,
            "direction": 0.05,
            "acyclicity": 0.05,
            "continuity": 0.08,
        },
        "interactions": {
            ("outcome", "direction"): 0.06,
            ("outcome", "acyclicity"): 0.06,
            ("direction", "continuity"): 0.065,
            ("acyclicity", "continuity"): 0.065,
            ("direction", "acyclicity"): 0.10,
        },
    },
    "structure_forward": {
        "singletons": {
            "outcome": 0.40,
            "format": 0.05,
            "direction": 0.06,
            "acyclicity": 0.06,
            "continuity": 0.08,
        },
        "interactions": {
            ("outcome", "direction"): 0.06,
            ("outcome", "acyclicity"): 0.06,
            ("direction", "continuity"): 0.065,
            ("acyclicity", "continuity"): 0.065,
            ("direction", "acyclicity"): 0.10,
        },
    },
}


def json_capacity_profiles() -> dict[str, dict[str, object]]:
    """Return JSON-safe profiles without changing the training representation."""
    return {
        name: {
            "singletons": dict(profile["singletons"]),
            "interactions": [
                {"sources": [left, right], "weight": weight}
                for (left, right), weight in profile["interactions"].items()
            ],
        }
        for name, profile in CHOQUET_CAPACITY_PROFILES.items()
    }
