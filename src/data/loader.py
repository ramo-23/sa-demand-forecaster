import codecs
import csv
from io import StringIO
import logging
from pathlib import Path

import pandas as pd

TIMEZONE = "Africa/Johannesburg"
LOGGER = logging.getLogger(__name__)
SUPPORTED_DELIMITERS = ",;\t|"


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


def _detect_delimiter(text: str, path: Path) -> str:
    sample = "\n".join(text.splitlines()[:20])
    try:
        return csv.Sniffer().sniff(sample, delimiters=SUPPORTED_DELIMITERS).delimiter
    except csv.Error as error:
        raise ValueError(f"Could not detect the delimiter in {path}") from error


def _parse_raw(path: str | Path) -> pd.DataFrame:
    """PLACEHOLDER: schema unknown until real Eskom file inspected."""
    source = Path(path)
    text = _decode_file(source)
    delimiter = _detect_delimiter(text, source)
    return pd.read_csv(StringIO(text), sep=delimiter)


def _identify_columns(frame: pd.DataFrame, path: Path) -> tuple[str, str]:
    datetime_candidates: list[tuple[str, pd.Series]] = []
    for column in frame.columns:
        if pd.api.types.is_numeric_dtype(frame[column]):
            continue
        parsed = pd.to_datetime(frame[column], errors="coerce", dayfirst=True)
        if len(parsed) and parsed.notna().all():
            datetime_candidates.append((column, parsed))

    if len(datetime_candidates) != 1:
        candidates = [str(column) for column, _ in datetime_candidates]
        raise ValueError(
            f"Could not uniquely identify the timestamp column in {path}; "
            f"datetime candidates: {candidates}. Inspect the Eskom CSV schema."
        )

    timestamp_column = datetime_candidates[0][0]
    numeric_candidates = []
    for column in frame.columns:
        if column == timestamp_column:
            continue
        parsed = pd.to_numeric(frame[column], errors="coerce")
        populated = frame[column].notna()
        if len(parsed) and populated.any() and parsed[populated].notna().all():
            numeric_candidates.append(column)

    if len(numeric_candidates) != 1:
        raise ValueError(
            f"Could not uniquely identify the demand column in {path}; "
            f"numeric candidates: {[str(column) for column in numeric_candidates]}. "
            "Inspect the Eskom CSV schema."
        )
    return timestamp_column, numeric_candidates[0]


def load_demand(path: str | Path) -> pd.DataFrame:
    """Load demand observations into a timezone-aware hourly frame.

    Naive source timestamps are interpreted as local South African time.
    Missing hourly timestamps remain NaN and are reported through the logger.
    """
    source = Path(path)
    raw = _parse_raw(source)
    timestamp_column, demand_column = _identify_columns(raw, source)

    timestamps = pd.DatetimeIndex(
        pd.to_datetime(raw[timestamp_column], errors="raise", dayfirst=True)
    )
    if timestamps.tz is None:
        timestamps = timestamps.tz_localize(
            TIMEZONE, ambiguous="raise", nonexistent="raise"
        )
    else:
        timestamps = timestamps.tz_convert(TIMEZONE)
    timestamps.name = "timestamp"

    if (
        (timestamps.minute != 0).any()
        or (timestamps.second != 0).any()
        or (timestamps.microsecond != 0).any()
        or (timestamps.nanosecond != 0).any()
    ):
        raise ValueError("Demand timestamps must be aligned to whole hours")

    observations = pd.DataFrame(
        {"demand_mw": pd.to_numeric(raw[demand_column], errors="raise").to_numpy()},
        index=timestamps,
    )
    observations = observations.sort_index(kind="mergesort")
    observations = observations.loc[~observations.index.duplicated(keep="first")]

    if observations.empty:
        raise ValueError(f"No demand observations found in {source}")

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
            "Demand data in %s is missing %d hourly timestamps: %s",
            source,
            len(missing_hours),
            ", ".join(timestamp.isoformat() for timestamp in missing_hours),
        )

    result = observations.reindex(full_index)
    result.index.name = "timestamp"
    return result
