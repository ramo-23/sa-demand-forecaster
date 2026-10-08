from collections.abc import Sequence

import numpy as np


def _as_matching_arrays(
    actual: Sequence[float] | np.ndarray,
    predicted: Sequence[float] | np.ndarray,
    actual_name: str,
    predicted_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    actual_values = np.asarray(actual, dtype=float)
    predicted_values = np.asarray(predicted, dtype=float)
    if actual_values.ndim != 1 or predicted_values.ndim != 1:
        raise ValueError("metric inputs must be one-dimensional")
    if actual_values.size == 0 or actual_values.shape != predicted_values.shape:
        raise ValueError(
            f"{actual_name} and {predicted_name} must have the same non-zero length"
        )
    if not np.isfinite(actual_values).all() or not np.isfinite(predicted_values).all():
        raise ValueError("metric inputs must contain only finite values")
    return actual_values, predicted_values


def mape(
    y_true: Sequence[float] | np.ndarray,
    y_pred: Sequence[float] | np.ndarray,
) -> float:
    """Return mean absolute percentage error as a fraction, not a percentage."""
    actual, predicted = _as_matching_arrays(y_true, y_pred, "y_true", "y_pred")
    if (actual == 0).any():
        raise ValueError("MAPE is undefined when y_true contains zero")
    return float(np.mean(np.abs((actual - predicted) / actual)))


def mase(
    y_true: Sequence[float] | np.ndarray,
    y_pred: Sequence[float] | np.ndarray,
    y_train: Sequence[float] | np.ndarray,
    season: int = 168,
) -> float:
    """Return MASE using the training series' seasonal-naive MAE as scale.

    MASE = mean(|y_true - y_pred|) /
    mean(|y_train[t] - y_train[t - season]|) for t = season, ..., n_train - 1.
    The scale is computed only from the in-sample training series.
    """
    actual, predicted = _as_matching_arrays(y_true, y_pred, "y_true", "y_pred")
    training = np.asarray(y_train, dtype=float)
    if training.ndim != 1 or not np.isfinite(training).all():
        raise ValueError("y_train must be one-dimensional and contain finite values")
    if not isinstance(season, int) or season <= 0:
        raise ValueError("season must be a positive integer")
    if len(training) <= season:
        raise ValueError("y_train must contain more values than the season length")

    forecast_error = float(np.mean(np.abs(actual - predicted)))
    scale = float(np.mean(np.abs(training[season:] - training[:-season])))
    if scale == 0:
        if forecast_error == 0:
            return 0.0
        raise ValueError("MASE is undefined when the seasonal-naive scale is zero")
    return forecast_error / scale


def interval_coverage(
    y_true: Sequence[float] | np.ndarray,
    lower: Sequence[float] | np.ndarray,
    upper: Sequence[float] | np.ndarray,
) -> float:
    """Return the fraction of observations contained in their prediction intervals."""
    actual, lower_values = _as_matching_arrays(y_true, lower, "y_true", "lower")
    upper_values = np.asarray(upper, dtype=float)
    if upper_values.ndim != 1 or upper_values.shape != actual.shape:
        raise ValueError("upper and y_true must have the same non-zero length")
    if not np.isfinite(upper_values).all():
        raise ValueError("metric inputs must contain only finite values")
    if (lower_values > upper_values).any():
        raise ValueError("lower bounds must not exceed upper bounds")
    return float(np.mean((actual >= lower_values) & (actual <= upper_values)))


def mean_interval_width(
    lower: Sequence[float] | np.ndarray,
    upper: Sequence[float] | np.ndarray,
) -> float:
    """Return the mean width of prediction intervals."""
    lower_values, upper_values = _as_matching_arrays(
        lower, upper, "lower", "upper"
    )
    if (lower_values > upper_values).any():
        raise ValueError("lower bounds must not exceed upper bounds")
    return float(np.mean(upper_values - lower_values))
