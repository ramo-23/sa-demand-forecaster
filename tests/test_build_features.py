import numpy as np
import pandas as pd
import pytest

from src.features.build import build_features

TIMEZONE = "Africa/Johannesburg"


def _inputs(length: int = 900) -> tuple[pd.DataFrame, pd.DataFrame]:
    index = pd.date_range(
        "2024-01-01",
        periods=length,
        freq="h",
        tz=TIMEZONE,
        name="timestamp",
    )
    demand = pd.DataFrame(
        {"demand_mw": 25000.0 + np.arange(length, dtype=float)},
        index=index,
    )
    weather = pd.DataFrame(
        {
            "temp_weighted": 18.0 + np.sin(np.arange(length, dtype=float) / 24),
            "temp_johannesburg": 19.0 + np.sin(np.arange(length, dtype=float) / 24),
        },
        index=index,
    )
    return demand, weather


def test_build_features_combines_weather_calendar_lags_and_rollings() -> None:
    demand, weather = _inputs()

    features = build_features(demand, weather)

    assert features.index.equals(demand.index)
    assert features.index.name == "timestamp"
    assert {"temp_weighted", "temp_johannesburg"}.issubset(features.columns)
    assert {
        "hour",
        "dayofweek",
        "month",
        "is_weekend",
        "is_public_holiday",
        "is_bridge_day",
    }.issubset(features.columns)
    assert {
        "demand_lag_24",
        "demand_lag_48",
        "demand_lag_168",
        "demand_lag_336",
        "demand_rolling_mean_24",
        "demand_rolling_std_24",
        "demand_rolling_mean_168",
        "demand_rolling_std_168",
    }.issubset(features.columns)
    assert "demand_mw" not in features.columns


def test_lag_values_at_horizon_24_have_expected_shifts() -> None:
    demand, weather = _inputs()
    demand["demand_mw"] = np.arange(len(demand), dtype=float)
    features = build_features(demand, weather, horizon=24)
    positions = np.arange(len(features), dtype=float)

    for lag in (24, 48, 168, 336):
        column = features[f"demand_lag_{lag}"]
        valid = column.notna().to_numpy()
        np.testing.assert_array_equal(
            positions[valid] - column.to_numpy()[valid],
            np.full(valid.sum(), lag, dtype=float),
        )


def test_lag_values_at_horizon_48_have_expected_shifts() -> None:
    demand, weather = _inputs()
    demand["demand_mw"] = np.arange(len(demand), dtype=float)
    features = build_features(demand, weather, horizon=48)
    positions = np.arange(len(features), dtype=float)

    expected_shifts = {
        24: 48,
        48: 72,
        168: 192,
        336: 360,
    }
    for lag, expected_shift in expected_shifts.items():
        column = features[f"demand_lag_{lag}"]
        valid = column.notna().to_numpy()
        np.testing.assert_array_equal(
            positions[valid] - column.to_numpy()[valid],
            np.full(valid.sum(), expected_shift, dtype=float),
        )


@pytest.mark.parametrize("horizon", [24, 48])
def test_perturbation_cannot_change_rows_through_horizon_minus_one(
    horizon: int,
) -> None:
    demand, weather = _inputs()
    perturbed = demand.copy()
    perturb_timestamp = demand.index[500]
    perturbed.loc[perturb_timestamp, "demand_mw"] += 10000

    original_features = build_features(demand, weather, horizon=horizon)
    perturbed_features = build_features(perturbed, weather, horizon=horizon)
    unaffected_end = perturb_timestamp + pd.Timedelta(hours=horizon - 1)

    pd.testing.assert_frame_equal(
        original_features.loc[:unaffected_end],
        perturbed_features.loc[:unaffected_end],
    )


@pytest.mark.parametrize("horizon", [24, 48])
def test_perturbation_does_change_rows_from_horizon_onward(horizon: int) -> None:
    demand, weather = _inputs()
    perturbed = demand.copy()
    perturb_timestamp = demand.index[500]
    perturbed.loc[perturb_timestamp, "demand_mw"] += 10000

    original_features = build_features(demand, weather, horizon=horizon)
    perturbed_features = build_features(perturbed, weather, horizon=horizon)
    start = perturb_timestamp + pd.Timedelta(hours=horizon)
    end = start + pd.Timedelta(hours=23)

    assert not original_features.loc[start:end].equals(
        perturbed_features.loc[start:end]
    )


def test_rejects_weather_that_does_not_cover_demand_index() -> None:
    demand, weather = _inputs()

    with pytest.raises(ValueError, match="every demand timestamp"):
        build_features(demand, weather.iloc[1:])


def test_rejects_non_positive_horizon() -> None:
    demand, weather = _inputs()

    with pytest.raises(ValueError, match="horizon"):
        build_features(demand, weather, horizon=0)