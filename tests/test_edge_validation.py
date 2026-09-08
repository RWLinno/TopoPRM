import pytest

from rebuttal.scripts.edge_validation import _source_aggregation_diagnostic


def test_source_aggregation_diagnostic_detects_cancellation():
    metrics = {}
    for index, (direction, acyclicity) in enumerate(
        [(0.1, 0.9), (0.2, 0.8), (0.3, 0.7), (0.4, 0.6)]
    ):
        metrics[str(index)] = {
            "original": {"direction_score": 0.9, "acyclicity_score": 0.5},
            "reversed": {
                "direction_score": direction,
                "acyclicity_score": acyclicity,
            },
        }

    result = _source_aggregation_diagnostic(metrics, reps=100, seed=0)

    assert result["n"] == 4
    assert result["correlation"]["value"] == pytest.approx(-1.0)
    assert result["average_std"]["value"] == pytest.approx(0.0)
    assert result["average_to_direction_std_ratio"]["value"] == pytest.approx(0.0)
    assert result["opposite_delta_fraction"]["value"] == pytest.approx(1.0)
