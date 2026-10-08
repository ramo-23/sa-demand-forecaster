import logging
from pathlib import Path

import pandas as pd
import pytest

from src.data.loader import ESKOM_COLUMNS, TIMEZONE, load_demand, load_eskom

HEADER = (
    "Date Time Hour Beginning,Residual Forecast,RSA Contracted Forecast,"
    "Dispatchable Generation,Residual Demand,RSA Contracted Demand,"
    "Thermal Generation,ILS Usage,Manual Load_Reduction(MLR),Wind,PV,"
    "Total RE,Installed Eskom Capacity,Total UCLF+OCLF\n"
)


def _row(timestamp: str, final_fields: tuple[str, ...] = ("999",)) -> str:
    values = [
        "1",
        "2",
        "3",
        "4",
        "500",
        "6",
        "7",
        "8",
        "9",
        "10",
        "11",
        "12",
    ]
    return ",".join([timestamp, *values, *final_fields]) + "\n"


def _write_fixture(path: Path, rows: str) -> Path:
    path.write_text(HEADER + rows, encoding="utf-8")
    return path


def test_load_eskom_parses_decimal_comma_integer_tail_and_midday(
    tmp_path: Path,
) -> None:
    path = _write_fixture(
        tmp_path / "eskom.csv",
        _row("2022-04-01 12:00:00 AM", ("12641", "035"))
        + _row("2022-04-01 01:00:00 AM", ("12642",))
        + _row("2022-04-01 12:00:00 PM", ("12643",)),
    )

    eskom = load_eskom(path)

    assert eskom.columns.tolist() == list(ESKOM_COLUMNS)
    assert eskom.index.name == "timestamp"
    assert str(pd.DatetimeIndex(eskom.index).tz) == TIMEZONE
    assert eskom.index[0].hour == 0
    assert eskom.index[-1].hour == 12
    assert eskom.loc["2022-04-01 00:00", "uclf_oclf"] == pytest.approx(12641.035)
    assert eskom.loc["2022-04-01 01:00", "uclf_oclf"] == 12642.0
    assert eskom.loc["2022-04-01 12:00", "uclf_oclf"] == 12643.0


def test_load_eskom_keeps_forecast_only_rows(tmp_path: Path) -> None:
    forecast_values = [
        "1",
        "2",
        "3",
        "4",
        "",
        "6",
        "7",
        "8",
        "9",
        "10",
        "11",
        "12",
    ]
    forecast_row = ",".join(
        ["2022-04-01 01:00:00 PM", *forecast_values, ""]
    ) + "\n"
    path = _write_fixture(
        tmp_path / "forecast.csv",
        _row("2022-04-01 12:00:00 PM") + forecast_row,
    )

    eskom = load_eskom(path)

    assert len(eskom) == 2
    assert pd.isna(eskom.iloc[-1]["rsa_contracted_demand"])
    assert pd.isna(eskom.iloc[-1]["uclf_oclf"])


def test_load_demand_drops_only_trailing_missing_demand_and_reports_count(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    forecast_values = [
        "1",
        "2",
        "3",
        "4",
        "",
        "6",
        "7",
        "8",
        "9",
        "10",
        "11",
        "12",
    ]
    forecast_row = ",".join(
        ["2022-04-01 02:00:00 AM", *forecast_values, ""]
    ) + "\n"
    path = _write_fixture(
        tmp_path / "demand.csv",
        _row("2022-04-01 12:00:00 AM") + _row("2022-04-01 01:00:00 AM") + forecast_row,
    )

    with caplog.at_level(logging.WARNING):
        demand = load_demand(path)

    assert demand.columns.tolist() == ["demand_mw"]
    assert demand["demand_mw"].tolist() == [500.0, 500.0]
    assert "Dropped 1 trailing Eskom rows" in caplog.text


def test_load_demand_raises_for_internal_missing_demand_with_timestamps(
    tmp_path: Path,
) -> None:
    forecast_values = [
        "1",
        "2",
        "3",
        "4",
        "",
        "6",
        "7",
        "8",
        "9",
        "10",
        "11",
        "12",
    ]
    forecast_row = ",".join(
        ["2022-04-01 01:00:00 AM", *forecast_values, ""]
    ) + "\n"
    path = _write_fixture(
        tmp_path / "internal_missing.csv",
        _row("2022-04-01 12:00:00 AM")
        + forecast_row
        + _row("2022-04-01 02:00:00 AM"),
    )

    with pytest.raises(ValueError, match="2022-04-01T01:00:00"):
        load_demand(path)


def test_duplicate_timestamps_keep_first_and_gaps_are_reported(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    path = _write_fixture(
        tmp_path / "gaps.csv",
        _row("2022-04-01 12:00:00 AM")
        + _row("2022-04-01 12:00:00 AM")
        + _row("2022-04-01 02:00:00 AM"),
    )

    with caplog.at_level(logging.WARNING):
        eskom = load_eskom(path)

    assert len(eskom) == 3
    assert eskom.iloc[0]["rsa_contracted_demand"] == 500.0
    assert pd.isna(eskom.iloc[1]["rsa_contracted_demand"])
    assert "missing 1 hourly timestamps" in caplog.text


def test_load_eskom_rejects_off_hour_timestamp(tmp_path: Path) -> None:
    path = _write_fixture(
        tmp_path / "off_grid.csv",
        _row("2022-04-01 12:30:00 AM"),
    )

    with pytest.raises(ValueError, match="aligned to whole hours"):
        load_eskom(path)


def test_invalid_trailing_field_count_reports_line_number(tmp_path: Path) -> None:
    path = tmp_path / "invalid.csv"
    invalid_row = _row("2022-04-01 12:00:00 AM", ("1", "2", "3"))
    path.write_text(HEADER + invalid_row, encoding="utf-8")

    with pytest.raises(ValueError, match="line 2"):
        load_eskom(path)
