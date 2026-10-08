import numpy as np
import pytest

from src.evaluate.metrics import interval_coverage, mase, mape, mean_interval_width
from src.models.baseline import seasonal_naive_forecast


def test_mase_is_zero_for_perfect_seasonal_naive_forecast() -> None:
    season = 4
    training = np.tile([10.0, 12.0, 14.0, 16.0], 3)
    actual = np.array([10.0, 12.0, 14.0, 16.0])
    predicted = seasonal_naive_forecast(training, len(actual), season=season)

    assert mase(actual, predicted, training, season=season) == 0.0


def test_mape_returns_fractional_error() -> None:
    assert mape([100.0, 200.0], [90.0, 220.0]) == pytest.approx(0.1)


def test_interval_coverage_for_wide_and_excluding_bounds() -> None:
    actual = [1.0, 2.0, 3.0]

    assert interval_coverage(actual, [-100.0] * 3, [100.0] * 3) == 1.0
    assert interval_coverage(actual, [4.0] * 3, [5.0] * 3) == 0.0


def test_mean_interval_width() -> None:
    assert mean_interval_width([0.0, 1.0], [2.0, 5.0]) == pytest.approx(3.0)
