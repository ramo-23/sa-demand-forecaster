import logging
from pathlib import Path

import pandas as pd
import pytest

from src.data.loader import TIMEZONE, load_demand


def _write_fixture(path: Path, contents: str) -> Path:
    path.write_text(contents, encoding="utf-8")
    return path


def test_load_demand_returns_sorted_contract_and_drops_duplicate_timestamps(
    tmp_path: Path,
) -> None:
    path = _write_fixture(
        tmp_path / "demand.tsv",
        "field_a\tfield_b\n"
        "2024-01-01 01:00\t101\n"
        "2024-01-01 00:00\t100\n"
        "2024-01-01 00:00\t999\n",
    )

    demand = load_demand(path)

    assert demand.columns.tolist() == ["demand_mw"]
    assert demand.index.name == "timestamp"
    assert str(pd.DatetimeIndex(demand.index).tz) == TIMEZONE
    assert demand.index.is_monotonic_increasing
    assert demand["demand_mw"].tolist() == [100, 101]


def test_load_demand_reports_gaps_and_keeps_them_missing(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    path = _write_fixture(
        tmp_path / "demand.txt",
        "field_a;field_b\n"
        "2024-01-01 00:00;100\n"
        "2024-01-01 02:00;102\n",
    )

    with caplog.at_level(logging.WARNING):
        demand = load_demand(path)

    assert len(demand) == 3
    assert pd.isna(demand.loc["2024-01-01 01:00", "demand_mw"])
    assert "missing 1 hourly timestamps" in caplog.text


def test_load_demand_rejects_timestamps_off_hour_grid(tmp_path: Path) -> None:
    path = _write_fixture(
        tmp_path / "demand.csv",
        "field_a,field_b\n"
        "2024-01-01 00:00,100\n"
        "2024-01-01 01:30,101\n",
    )

    with pytest.raises(ValueError, match="aligned to whole hours"):
        load_demand(path)


def test_load_demand_detects_windows_1252_encoding(tmp_path: Path) -> None:
    path = tmp_path / "demand.psv"
    path.write_bytes(
        "field_a|consommation\n2024-01-01 00:00|100\n".encode("cp1252")
    )

    demand = load_demand(path)

    assert demand.columns.tolist() == ["demand_mw"]
    assert demand["demand_mw"].tolist() == [100]
