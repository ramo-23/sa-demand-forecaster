import pandas as pd

from src.features.calendar import build_calendar_features

TIMEZONE = "Africa/Johannesburg"
DEMAND_COLUMN = "demand_mw"
DEMAND_LAGS = (24, 48, 168, 336)
ROLLING_WINDOWS = (24, 168)


def _validate_hourly_index(index: pd.Index, name: str) -> pd.DatetimeIndex:
    if not isinstance(index, pd.DatetimeIndex):
        raise TypeError(f"{name} must have a pandas DatetimeIndex")
    if index.tz is None or str(index.tz) != TIMEZONE:
        raise ValueError(f"{name} index must be timezone-aware in {TIMEZONE}")
    if index.has_duplicates or not index.is_monotonic_increasing:
        raise ValueError(f"{name} index must be increasing with no duplicates")
    if len(index) > 1:
        differences = index[1:] - index[:-1]
        if not (differences == pd.Timedelta(hours=1)).all():
            raise ValueError(f"{name} index must have consecutive hourly timestamps")
    return index


def build_features(
    demand_df: pd.DataFrame,
    weather_df: pd.DataFrame,
    horizon: int = 24,
) -> pd.DataFrame:
    """Build hourly weather, calendar, lag, and rolling demand predictors.

    Demand lag columns use their named offsets. Rolling features are computed
    from demand shifted by the forecast horizon, so the newest observation in
    a rolling window is no later than the forecast origin.

    demand_lag_k means k hours before the earliest permitted observation.
    Actual shift = k + (horizon - 24). At horizon=24 the shift equals k.
    """
    if not isinstance(horizon, int) or horizon <= 0:
        raise ValueError("horizon must be a positive integer")
    if DEMAND_COLUMN not in demand_df.columns:
        raise ValueError(f"demand_df must contain a {DEMAND_COLUMN!r} column")
    if demand_df.empty:
        raise ValueError("demand_df must not be empty")
    if weather_df.empty:
        raise ValueError("weather_df must not be empty")

    demand_index = _validate_hourly_index(demand_df.index, "demand_df")
    _validate_hourly_index(weather_df.index, "weather_df")
    if not weather_df.columns.is_unique:
        raise ValueError("weather_df columns must be unique")
    if DEMAND_COLUMN in weather_df.columns:
        raise ValueError(f"weather_df must not contain {DEMAND_COLUMN!r}")

    weather = weather_df.reindex(demand_index)
    if weather.isna().any().any():
        missing_columns = weather.columns[weather.isna().any()].tolist()
        raise ValueError(
            "weather_df must have non-missing values for every demand timestamp; "
            f"missing values in: {', '.join(map(str, missing_columns))}"
        )

    demand = pd.to_numeric(demand_df[DEMAND_COLUMN], errors="raise").astype(float)
    if demand.isna().any():
        raise ValueError(f"demand_df[{DEMAND_COLUMN!r}] must not contain missing values")

    features = weather.copy()
    calendar = build_calendar_features(demand_index)
    overlapping_columns = features.columns.intersection(calendar.columns)
    if len(overlapping_columns):
        raise ValueError(
            "weather_df columns conflict with calendar features: "
            f"{', '.join(map(str, overlapping_columns))}"
        )
    features = features.join(calendar)

    for lag in DEMAND_LAGS:
        horizon_adjustment = max(0, horizon - 24)
        features[f"demand_lag_{lag}"] = demand.shift(lag + horizon_adjustment)

    horizon_shifted_demand = demand.shift(horizon)
    for window in ROLLING_WINDOWS:
        rolling = horizon_shifted_demand.rolling(window=window, min_periods=window)
        features[f"demand_rolling_mean_{window}"] = rolling.mean()
        features[f"demand_rolling_std_{window}"] = rolling.std()

    features.index.name = "timestamp"
    return features
