from datetime import date

import numpy as np
import pandas as pd

from src.features.calendar import build_calendar_features
from src.features.weather import fetch_weather

TIMEZONE = "Africa/Johannesburg"


def _as_local_timestamp(value: str | date | pd.Timestamp, name: str) -> pd.Timestamp:
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a valid date or timestamp") from error
    if pd.isna(timestamp):
        raise ValueError(f"{name} must be a valid date or timestamp")
    if timestamp.tz is None:
        return timestamp.tz_localize(TIMEZONE, ambiguous="raise", nonexistent="raise")
    return timestamp.tz_convert(TIMEZONE)


def make_synthetic_demand(
    start: str | date | pd.Timestamp,
    end: str | date | pd.Timestamp,
    seed: int = 42,
) -> pd.DataFrame:
    """TESTING ONLY, results meaningless."""
    start_timestamp = _as_local_timestamp(start, "start")
    end_timestamp = _as_local_timestamp(end, "end")

    if isinstance(end, date) and not isinstance(end, pd.Timestamp):
        end_timestamp += pd.Timedelta(hours=23)
    elif isinstance(end, str) and len(end) == 10:
        end_timestamp += pd.Timedelta(hours=23)

    if end_timestamp < start_timestamp:
        raise ValueError("end must be on or after start")
    if start_timestamp.minute or start_timestamp.second or start_timestamp.microsecond:
        raise ValueError("start must be aligned to a whole hour")
    if end_timestamp.minute or end_timestamp.second or end_timestamp.microsecond:
        raise ValueError("end must be aligned to a whole hour")

    index = pd.date_range(
        start=start_timestamp,
        end=end_timestamp,
        freq="h",
        tz=TIMEZONE,
        name="timestamp",
    )
    if index.empty:
        raise ValueError("start and end must contain at least one hourly timestamp")

    weather = fetch_weather(index[0].date().isoformat(), index[-1].date().isoformat())
    weather = weather.reindex(index)
    if "temp_weighted" not in weather.columns:
        raise ValueError("Weather data must contain the temp_weighted column")
    if weather["temp_weighted"].isna().any():
        raise ValueError("Weather data is incomplete over the requested period")

    calendar = build_calendar_features(index)
    hour_angle = 2 * np.pi * (index.hour.to_numpy() - 16) / 24
    daily_pattern = 1700 * np.cos(hour_angle) + 350 * np.cos(2 * hour_angle)
    weekly_pattern = np.where(calendar["dayofweek"].to_numpy() < 5, 550, -850)
    holiday_effect = -1200 * calendar["is_public_holiday"].to_numpy(dtype=float)
    temperature_effect = 65 * (
        weather["temp_weighted"].to_numpy(dtype=float) - 18
    )
    noise = np.random.default_rng(seed).normal(0, 300, size=len(index))

    demand = (
        30000
        + daily_pattern
        + weekly_pattern
        + holiday_effect
        + temperature_effect
        + noise
    )
    return pd.DataFrame({"demand_mw": demand}, index=index)
