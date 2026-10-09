from collections.abc import Sequence
import math

import numpy as np


def conformalize_intervals(
    cal_y: Sequence[float] | np.ndarray,
    cal_lower: Sequence[float] | np.ndarray,
    cal_upper: Sequence[float] | np.ndarray,
    test_lower: Sequence[float] | np.ndarray,
    test_upper: Sequence[float] | np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply split-conformal CQR to calibration and test intervals.

    The finite-sample correction is an upper-rounded empirical quantile of
    ``max(cal_lower - cal_y, cal_y - cal_upper)``. The time-series coverage
    guarantee relies on exchangeability, which generally does not hold for
    serially dependent electricity-demand observations; coverage is therefore
    not guaranteed for this forecasting setting.
    """
    calibration_y = np.asarray(cal_y, dtype=float)
    calibration_lower = np.asarray(cal_lower, dtype=float)
    calibration_upper = np.asarray(cal_upper, dtype=float)
    test_lower_values = np.asarray(test_lower, dtype=float)
    test_upper_values = np.asarray(test_upper, dtype=float)
    calibration_arrays = (
        calibration_y,
        calibration_lower,
        calibration_upper,
    )
    if any(values.ndim != 1 for values in calibration_arrays):
        raise ValueError("calibration inputs must be one-dimensional")
    if not calibration_y.size or any(
        values.shape != calibration_y.shape for values in calibration_arrays
    ):
        raise ValueError("calibration inputs must have the same non-zero length")
    if (
        test_lower_values.ndim != 1
        or test_upper_values.ndim != 1
        or test_lower_values.shape != test_upper_values.shape
    ):
        raise ValueError("test interval bounds must be matching one-dimensional arrays")
    if not all(np.isfinite(values).all() for values in (*calibration_arrays,
                                                        test_lower_values,
                                                        test_upper_values)):
        raise ValueError("conformal inputs must contain only finite values")
    if np.any(calibration_lower > calibration_upper):
        raise ValueError("calibration lower bounds must not exceed upper bounds")
    if np.any(test_lower_values > test_upper_values):
        raise ValueError("test lower bounds must not exceed upper bounds")
    if not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must be strictly between 0 and 1")

    scores = np.maximum(
        calibration_lower - calibration_y,
        calibration_y - calibration_upper,
    )
    quantile_level = min(
        1.0,
        math.ceil((len(scores) + 1) * (1.0 - alpha)) / len(scores),
    )
    correction = float(np.quantile(scores, quantile_level, method="higher"))
    return test_lower_values - correction, test_upper_values + correction
