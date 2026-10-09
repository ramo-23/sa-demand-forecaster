import numpy as np
import pytest

from src.models.conformal import conformalize_intervals


def test_correction_can_be_negative_when_calibration_points_are_well_inside() -> None:
    cal_y = np.zeros(20)
    cal_lower = np.full(20, -10.0)
    cal_upper = np.full(20, 10.0)

    lower, upper = conformalize_intervals(
        cal_y,
        cal_lower,
        cal_upper,
        np.array([-2.0]),
        np.array([2.0]),
        alpha=0.2,
    )

    assert lower[0] > -2.0
    assert upper[0] < 2.0


def test_correction_is_positive_and_larger_for_points_outside_intervals() -> None:
    cal_y = np.array([-5.0, 5.0, -4.0, 4.0])
    cal_lower = np.full(4, -1.0)
    cal_upper = np.full(4, 1.0)

    lower, upper = conformalize_intervals(
        cal_y,
        cal_lower,
        cal_upper,
        np.array([-1.0]),
        np.array([1.0]),
        alpha=0.25,
    )

    assert lower[0] < -1.0
    assert upper[0] > 1.0
    assert upper[0] - 1.0 > 2.0


def test_conformal_coverage_is_closer_to_nominal_for_narrow_intervals() -> None:
    calibration_y = np.linspace(-10.0, 10.0, 1000)
    test_y = np.linspace(-10.0, 10.0, 1000)
    calibration_lower = np.full_like(calibration_y, -1.0)
    calibration_upper = np.full_like(calibration_y, 1.0)
    test_lower = np.full_like(test_y, -1.0)
    test_upper = np.full_like(test_y, 1.0)

    raw_coverage = np.mean((test_y >= test_lower) & (test_y <= test_upper))
    conformal_lower, conformal_upper = conformalize_intervals(
        calibration_y,
        calibration_lower,
        calibration_upper,
        test_lower,
        test_upper,
        alpha=0.1,
    )
    conformal_coverage = np.mean(
        (test_y >= conformal_lower) & (test_y <= conformal_upper)
    )

    assert abs(conformal_coverage - 0.9) < abs(raw_coverage - 0.9)
    assert conformal_coverage == pytest.approx(0.9, abs=0.01)


def test_rejects_invalid_alpha_and_mismatched_shapes() -> None:
    with pytest.raises(ValueError, match="alpha"):
        conformalize_intervals([1.0], [0.0], [2.0], [0.0], [2.0], alpha=1.0)
    with pytest.raises(ValueError, match="same non-zero length"):
        conformalize_intervals([1.0], [0.0, 1.0], [2.0], [0.0], [2.0], alpha=0.1)
