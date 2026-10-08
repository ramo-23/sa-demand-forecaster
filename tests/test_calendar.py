import pandas as pd
import pytest

from src.features.calendar import build_calendar_features

TIMEZONE = "Africa/Johannesburg"
EXPECTED_COLUMNS = [
    "hour",
    "dayofweek",
    "month",
    "is_weekend",
    "is_public_holiday",
    "is_bridge_day",
]


def test_known_south_african_holidays_are_flagged() -> None:
    index = pd.date_range("2024-04-27", "2024-04-29 23:00", freq="h", tz=TIMEZONE)
    christmas_index = pd.date_range("2024-12-25", periods=24, freq="h", tz=TIMEZONE)

    features = build_calendar_features(index)

    assert features.loc["2024-04-27", "is_public_holiday"].all()
    assert not features.loc["2024-04-29", "is_public_holiday"].any()
    assert build_calendar_features(christmas_index)["is_public_holiday"].all()


def test_sunday_holiday_is_observed_on_monday() -> None:
    index = pd.date_range("2024-06-16", "2024-06-17 23:00", freq="h", tz=TIMEZONE)

    features = build_calendar_features(index)

    assert features.loc["2024-06-16", "is_public_holiday"].all()
    assert features.loc["2024-06-17", "is_public_holiday"].all()


def test_one_off_election_holiday_is_flagged() -> None:
    index = pd.date_range("2024-05-29", periods=24, freq="h", tz=TIMEZONE)

    assert build_calendar_features(index)["is_public_holiday"].all()


def test_weekends_and_bridge_days_are_flagged() -> None:
    index = pd.date_range("2024-12-26", "2024-12-29 23:00", freq="h", tz=TIMEZONE)

    features = build_calendar_features(index)

    assert features.loc["2024-12-28", "is_weekend"].all()
    assert features.loc["2024-12-29", "is_weekend"].all()
    assert features.loc["2024-12-27", "is_bridge_day"].all()
    assert not features.loc["2024-12-27", "is_weekend"].any()


def test_column_set_index_and_values_are_exact_and_complete() -> None:
    index = pd.date_range("2024-12-24", periods=72, freq="h", tz=TIMEZONE)

    features = build_calendar_features(index)

    assert features.columns.tolist() == EXPECTED_COLUMNS
    assert features.index.equals(index)
    assert not features.isna().any().any()


def test_rejects_naive_or_non_hourly_index() -> None:
    naive_index = pd.date_range("2024-01-01", periods=2, freq="h")
    non_hourly_index = pd.DatetimeIndex(
        ["2024-01-01 00:00", "2024-01-01 02:00"],
        tz=TIMEZONE,
    )

    with pytest.raises(ValueError, match="timezone-aware"):
        build_calendar_features(naive_index)
    with pytest.raises(ValueError, match="hourly"):
        build_calendar_features(non_hourly_index)
