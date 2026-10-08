import pandas as pd
import holidays

TIMEZONE = "Africa/Johannesburg"

CALENDAR_COLUMNS = [
    "hour",
    "dayofweek",
    "month",
    "is_weekend",
    "is_public_holiday",
    "is_bridge_day",
]


def build_calendar_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    if not isinstance(index, pd.DatetimeIndex):
        raise TypeError("index must be a pandas DatetimeIndex")
    if index.tz is None or str(index.tz) != TIMEZONE:
        raise ValueError(f"index must be timezone-aware in {TIMEZONE}")
    if not index.is_monotonic_increasing or index.has_duplicates:
        raise ValueError("index must be monotonic increasing with no duplicates")
    if len(index) > 1:
        differences = index[1:] - index[:-1]
        if not (differences == pd.Timedelta(hours=1)).all():
            raise ValueError("index must have hourly frequency")

    years = range(index.year.min() - 1, index.year.max() + 2) if len(index) else []
    public_holidays = holidays.country_holidays(
        "ZA",
        years=years,
        observed=True,
    )
    dates = index.date
    is_weekend = index.dayofweek >= 5
    is_public_holiday = pd.Series(
        [day in public_holidays for day in dates],
        index=index,
        dtype=bool,
    )

    previous_is_weekend = pd.Series(
        ((index.dayofweek - 1) % 7) >= 5,
        index=index,
    )
    next_is_weekend = pd.Series(
        ((index.dayofweek + 1) % 7) >= 5,
        index=index,
    )
    previous_is_holiday = pd.Series(
        [(day - pd.Timedelta(days=1)) in public_holidays for day in dates],
        index=index,
    )
    next_is_holiday = pd.Series(
        [(day + pd.Timedelta(days=1)) in public_holidays for day in dates],
        index=index,
    )
    is_bridge_day = (
        (previous_is_weekend & next_is_holiday)
        | (previous_is_holiday & next_is_weekend)
    ) & ~is_public_holiday

    return pd.DataFrame(
        {
            "hour": index.hour,
            "dayofweek": index.dayofweek,
            "month": index.month,
            "is_weekend": is_weekend,
            "is_public_holiday": is_public_holiday,
            "is_bridge_day": is_bridge_day,
        },
        index=index,
        columns=CALENDAR_COLUMNS,
    )
