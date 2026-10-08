from datetime import date
from pathlib import Path

import pandas as pd
import requests

API_URL = "https://archive-api.open-meteo.com/v1/archive"
TIMEZONE = "Africa/Johannesburg"

CITIES: dict[str, tuple[float, float]] = {
    "johannesburg": (-26.2041, 28.0473),
    "cape_town": (-33.9249, 18.4241),
    "durban": (-29.8587, 31.0218),
    "bloemfontein": (-29.0852, 26.1596),
}

WEIGHTS: dict[str, float] = {
    "johannesburg": 0.50,
    "durban": 0.20,
    "cape_town": 0.20,
    "bloemfontein": 0.10,
}

CITY_COLUMNS = {
    city: f"temp_{city}"
    for city in ("johannesburg", "cape_town", "durban", "bloemfontein")
}


def _parse_date(value: str, name: str) -> date:
    try:
        parsed = date.fromisoformat(value)
    except ValueError as error:
        raise ValueError(f"{name} must use YYYY-MM-DD format; got {value!r}") from error
    if parsed.isoformat() != value:
        raise ValueError(f"{name} must use YYYY-MM-DD format; got {value!r}")
    return parsed


def _fetch_city_year(
    city: str,
    lat: float,
    lon: float,
    start_date: date,
    end_date: date,
    session: requests.Session,
) -> pd.DataFrame:
    response = session.get(
        API_URL,
        params={
            "latitude": lat,
            "longitude": lon,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "hourly": "temperature_2m",
            "timezone": TIMEZONE,
        },
    )
    if response.status_code != 200:
        raise requests.HTTPError(
            f"Open-Meteo request failed for {city} from {start_date} "
            f"through {end_date}: HTTP {response.status_code}. "
            f"Response body: {response.text}",
            response=response,
        )

    payload = response.json()
    try:
        hourly = payload["hourly"]
        times = hourly["time"]
        temperatures = hourly["temperature_2m"]
    except (KeyError, TypeError) as error:
        raise ValueError(
            f"Open-Meteo response for {city} from {start_date} through "
            f"{end_date} is missing hourly time or temperature_2m data"
        ) from error

    if len(times) != len(temperatures):
        raise ValueError(
            f"Open-Meteo returned mismatched timestamps and temperatures "
            f"for {city} from {start_date} through {end_date}"
        )

    index = pd.DatetimeIndex(pd.to_datetime(times, errors="raise"))
    if index.tz is None:
        index = index.tz_localize(TIMEZONE, ambiguous="raise", nonexistent="raise")
    else:
        index = index.tz_convert(TIMEZONE)
    index.name = "timestamp"

    values = pd.to_numeric(pd.Series(temperatures), errors="raise")
    frame = pd.DataFrame({CITY_COLUMNS[city]: values.to_numpy()}, index=index)
    if frame.index.has_duplicates:
        raise ValueError(
            f"Open-Meteo returned duplicate timestamps for {city} "
            f"from {start_date} through {end_date}"
        )
    if not frame.index.is_monotonic_increasing:
        raise ValueError(
            f"Open-Meteo returned non-monotonic timestamps for {city} "
            f"from {start_date} through {end_date}"
        )
    return frame


def fetch_weather(
    start_date: str,
    end_date: str,
    cache_dir: Path = Path("data/raw"),
    session: requests.Session | None = None,
    force_refresh: bool = False,
) -> pd.DataFrame:
    start = _parse_date(start_date, "start_date")
    end = _parse_date(end_date, "end_date")
    if end < start:
        raise ValueError("end_date must be on or after start_date")

    output_columns = [
        "temp_johannesburg",
        "temp_cape_town",
        "temp_durban",
        "temp_bloemfontein",
        "temp_weighted",
    ]
    full_index = pd.date_range(
        start=start.isoformat(),
        end=f"{end.isoformat()} 23:00",
        freq="h",
        tz=TIMEZONE,
        name="timestamp",
    )
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    client = session if session is not None else requests.Session()
    owns_session = session is None
    city_frames: dict[str, pd.DataFrame] = {}
    try:
        for city, (latitude, longitude) in CITIES.items():
            column = CITY_COLUMNS[city]
            yearly_frames = []
            for year in range(start.year, end.year + 1):
                chunk_start = max(start, date(year, 1, 1))
                chunk_end = min(end, date(year, 12, 31))
                yearly_cache = cache_path / (
                    f"open_meteo_{city}_{chunk_start.isoformat()}_"
                    f"{chunk_end.isoformat()}.csv"
                )
                if yearly_cache.exists() and not force_refresh:
                    cached = pd.read_csv(yearly_cache, index_col="timestamp")
                    if column not in cached.columns:
                        raise ValueError(
                            f"Weather cache {yearly_cache} is missing column {column!r}"
                        )
                    cached_index = pd.DatetimeIndex(
                        pd.to_datetime(cached.index, errors="raise")
                    )
                    if cached_index.tz is None:
                        cached_index = cached_index.tz_localize(
                            TIMEZONE, ambiguous="raise", nonexistent="raise"
                        )
                    else:
                        cached_index = cached_index.tz_convert(TIMEZONE)
                    cached.index = cached_index
                    cached.index.name = "timestamp"
                    yearly = cached[[column]]
                else:
                    yearly = _fetch_city_year(
                        city,
                        latitude,
                        longitude,
                        chunk_start,
                        chunk_end,
                        client,
                    )
                    yearly.to_csv(yearly_cache, index_label="timestamp")
                yearly_frames.append(yearly)

            city_weather = pd.concat(yearly_frames).sort_index()
            city_frames[city] = city_weather.reindex(full_index)

        result = pd.concat(
            [city_frames[city] for city in CITY_COLUMNS],
            axis=1,
        )
        temperature_columns = list(CITY_COLUMNS.values())
        missing_rows = result[temperature_columns].isna().any(axis=1)
        if missing_rows.any():
            missing_timestamps = result.index[missing_rows]
            raise ValueError(
                "Weather data has missing temperatures at requested timestamps "
                f"from {missing_timestamps[0]} through {missing_timestamps[-1]}"
            )

        result["temp_weighted"] = sum(
            result[CITY_COLUMNS[city]] * weight
            for city, weight in WEIGHTS.items()
        )
        result = result[output_columns]
        result.index.name = "timestamp"
        return result
    finally:
        if owns_session:
            client.close()
