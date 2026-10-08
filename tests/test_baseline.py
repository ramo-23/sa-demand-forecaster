import numpy as np
import pytest

from src.models.baseline import seasonal_naive_forecast


def test_seasonal_naive_forecast_uses_only_latest_past_season() -> None:
    series = np.array([100, 101, 102, 103, 10, 11, 12, 13])

    forecast = seasonal_naive_forecast(series, horizon=6, season=4)

    np.testing.assert_array_equal(forecast, [10, 11, 12, 13, 10, 11])


def test_seasonal_naive_rejects_insufficient_history() -> None:
    with pytest.raises(ValueError, match="complete season"):
        seasonal_naive_forecast([1.0, 2.0], horizon=1, season=3)
