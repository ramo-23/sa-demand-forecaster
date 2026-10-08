import codecs
import csv
from datetime import datetime
from io import StringIO
import logging
from pathlib import Path

import pandas as pd

TIMEZONE = "Africa/Johannesburg"
LOGGER = logging.getLogger(__name__)
TIMESTAMP_FORMAT = "%Y-%m-%d %I:%M:%S %p"
ESKOM_HEADER = (
    "Date Time Hour Beginning",
    "Residual Forecast",
    "RSA Contracted Forecast",
    "Dispatchable Generation",
    "Residual Demand",
    "RSA Contracted Demand",
    "Thermal Generation",
    "ILS Usage",
    "Manual Load_Reduction(MLR)",
    "Wind",
    "PV",
    "Total RE",
    "Installed Eskom Capacity",
    "Total UCLF+OCLF",
)
ESKOM_COLUMNS = (
    "residual_forecast",
    "rsa_contracted_forecast",
    "dispatchable_generation",
    "residual_demand",
    "rsa_contracted_demand",
    "thermal_generation",
    "ils_usage",
    "mlr",
    "wind",
    "pv",
    "total_re",
    "installed_capacity",
    "uclf_oclf",
)


def _decode_file(path: Path) -> str:
    raw = path.read_bytes()
    if raw.startswith(codecs.BOM_UTF8):
        return raw.decode("utf-8-sig")
    if raw.startswith(codecs.BOM_UTF32_LE) or raw.startswith(codecs.BOM_UTF32_BE):
        return raw.decode("utf-32")
    if raw.startswith(codecs.BOM_UTF16_LE) or raw.startswith(codecs.BOM_UTF16_BE):
        return raw.decode("utf-16")
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("cp1252")


def _parse_number(value: str, line_number: int) -> float:
    if not value.strip():
        return float("nan")
    try:
        return float(value.strip())
    except ValueError as error:
        raise ValueError(
            f"Invalid numeric value on line {line_number}: {value!r}"
        ) from error


def _parse_raw(path: str | Path) -> pd.DataFrame:
    """Parse the verified Eskom export layout with its unquoted decimal comma."""
    source = Path(path)
    text = _decode_file(source)
    reader = csv.reader(StringIO(text, newline=""), delimiter=",")
    try:
        header = next(reader)
    except StopIteration as error:
        raise ValueError(f"Eskom file is empty: {source}") from error

    header = [field.strip() for field in header]
    if tuple(header) != ESKOM_HEADER:
        raise ValueError(
            f"Unexpected Eskom header on line 1 in {source}; "
            f"expected {len(ESKOM_HEADER)} verified fields"
        )

    timestamps: list[datetime] = []
    records: list[list[float]] = []
    for fields in reader:
        line_number = reader.line_num
        if not fields or (len(fields) == 1 and not fields[0].strip()):
            continue
        if len(fields) < 13:
            raise ValueError(
                f"Invalid field count on line {line_number}: "
                f"expected at least 14 fields, got {len(fields)}"
            )

        tail = fields[13:]
        if len(tail) == 2:
            last_value = _parse_number(
                f"{tail[0].strip()}.{tail[1].strip()}",
                line_number,
            )
        elif len(tail) == 1:
            last_value = _parse_number(tail[0], line_number)
        else:
            raise ValueError(
                f"Invalid trailing field count on line {line_number}: "
                f"expected 1 or 2 fields, got {len(tail)}"
            )

        try:
            timestamp = datetime.strptime(fields[0].strip(), TIMESTAMP_FORMAT)
        except ValueError as error:
            raise ValueError(
                f"Invalid timestamp on line {line_number}: {fields[0]!r}"
            ) from error

        try:
            values = [
                _parse_number(value, line_number)
                for value in fields[1:13]
            ]
        except ValueError:
            raise
        timestamps.append(timestamp)
        records.append([*values, last_value])

    if not records:
        raise ValueError(f"No Eskom data rows found in {source}")

    index = pd.DatetimeIndex(timestamps, name="timestamp")
    try:
        index = index.tz_localize(
            TIMEZONE,
            ambiguous="raise",
            nonexistent="raise",
        )
    except (ValueError, TypeError) as error:
        raise ValueError(
            "Could not localize Eskom timestamps to Africa/Johannesburg"
        ) from error

    if (
        (index.minute != 0).any()
        or (index.second != 0).any()
        or (index.microsecond != 0).any()
        or (index.nanosecond != 0).any()
    ):
        raise ValueError("Eskom timestamps must be aligned to whole hours")

    return pd.DataFrame(records, index=index, columns=ESKOM_COLUMNS)


def _sort_deduplicate_and_report_gaps(
    observations: pd.DataFrame,
    source: Path,
) -> pd.DataFrame:
    observations = observations.sort_index(kind="mergesort")
    observations = observations.loc[~observations.index.duplicated(keep="first")]
    if observations.empty:
        raise ValueError(f"No Eskom observations found in {source}")

    full_index = pd.date_range(
        start=observations.index[0],
        end=observations.index[-1],
        freq="h",
        tz=TIMEZONE,
        name="timestamp",
    )
    missing_hours = full_index.difference(observations.index)
    if len(missing_hours):
        LOGGER.warning(
            "Eskom data in %s is missing %d hourly timestamps: %s",
            source,
            len(missing_hours),
            ", ".join(timestamp.isoformat() for timestamp in missing_hours),
        )
    result = observations.reindex(full_index)
    result.index.name = "timestamp"
    return result


def load_eskom(path: str | Path) -> pd.DataFrame:
    """Load the verified Eskom hourly export into a timezone-aware frame.

    The file has no timezone information. Treating its timestamps as local
    Africa/Johannesburg time is an unverified assumption.
    """
    source = Path(path)
    return _sort_deduplicate_and_report_gaps(_parse_raw(source), source)


def load_demand(path: str | Path) -> pd.DataFrame:
    """Load RSA contracted demand into the hourly ``demand_mw`` contract.

    The Eskom file has no timezone information. Treating its timestamps as
    local Africa/Johannesburg time is an unverified assumption. Trailing rows
    without demand are dropped; missing demand at any earlier timestamp is an
    error. Interior time gaps remain NaN and are reported by ``load_eskom``.
    """
    eskom = load_eskom(path)
    demand = eskom["rsa_contracted_demand"]
    trailing_missing = demand.isna().iloc[::-1].cumprod().sum()
    if trailing_missing:
        LOGGER.warning(
            "Dropped %d trailing Eskom rows with missing RSA contracted demand",
            trailing_missing,
        )
        eskom = eskom.iloc[:-trailing_missing]
        demand = eskom["rsa_contracted_demand"]
    if demand.empty:
        raise ValueError("No rows with RSA contracted demand were found")

    missing_demand = demand[demand.isna()]
    if not missing_demand.empty:
        timestamps = ", ".join(
            timestamp.isoformat() for timestamp in missing_demand.index
        )
        raise ValueError(
            "RSA contracted demand is missing before the trailing rows at "
            f"timestamps: {timestamps}"
        )

    result = demand.to_frame(name="demand_mw")
    result.index.name = "timestamp"
    return result
