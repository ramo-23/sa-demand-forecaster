from datetime import date
import json
from pathlib import Path

import pandas as pd
import pytest
import requests

from src.features.weather import CITIES, WEIGHTS, fetch_weather


class FakeSession(requests.Session):
    def __init__(self, status_code: int = 200):
        super().__init__()
        self.status_code = status_code
        self.calls: list[tuple[str, dict]] = []

    def get(self, url: str, **kwargs: object) -> requests.Response:
        params = kwargs.get("params")
        if not isinstance(params, dict):
            raise AssertionError("Expected Open-Meteo request parameters")
        self.calls.append((url, params))
        year = int(params["start_date"][:4])
        start = date(year, 1, 1)
        end = date(year + 1, 1, 1)
        times = pd.date_range(start, end, freq="h", inclusive="left")
        response = requests.Response()
        response.status_code = self.status_code
        response._content = json.dumps(
            {
                "hourly": {
                    "time": times.strftime("%Y-%m-%dT%H:%M").tolist(),
                    "temperature_2m": [20.0] * len(times),
                }
            },
        ).encode("utf-8")
        return response


def test_weights_sum_to_one() -> None:
    assert sum(WEIGHTS.values()) == pytest.approx(1.0)


def test_returns_hourly_timezone_aware_wide_frame(tmp_path: Path) -> None:
    session = FakeSession()

    weather = fetch_weather(
        "2024-01-01",
        "2024-01-02",
        cache_dir=tmp_path,
        session=session,
    )

    assert weather.columns.tolist() == [
        "temp_johannesburg",
        "temp_cape_town",
        "temp_durban",
        "temp_bloemfontein",
        "temp_weighted",
    ]
    assert weather.index.name == "timestamp"
    index = pd.DatetimeIndex(weather.index)
    assert index.tz is not None
    assert str(index.tz) == "Africa/Johannesburg"
    assert index.is_monotonic_increasing
    assert len(weather) == 48
    assert weather["temp_weighted"].eq(20.0).all()
    assert len(session.calls) == len(CITIES)


def test_second_call_uses_yearly_cache_without_http(tmp_path: Path) -> None:
    session = FakeSession()

    first = fetch_weather("2024-01-01", "2024-01-01", tmp_path, session)
    call_count = len(session.calls)
    second = fetch_weather("2024-01-01", "2024-01-01", tmp_path, session)

    assert call_count == len(CITIES)
    assert len(session.calls) == call_count
    pd.testing.assert_frame_equal(first, second)


def test_force_refresh_re_fetches_all_cities(tmp_path: Path) -> None:
    session = FakeSession()

    fetch_weather("2024-01-01", "2024-01-01", tmp_path, session)
    fetch_weather(
        "2024-01-01",
        "2024-01-01",
        cache_dir=tmp_path,
        session=session,
        force_refresh=True,
    )

    assert len(session.calls) == len(CITIES) * 2


def test_non_200_response_raises_and_does_not_return_partial_data(
    tmp_path: Path,
) -> None:
    session = FakeSession(status_code=503)

    with pytest.raises(requests.HTTPError, match="HTTP 503"):
        fetch_weather(
            "2024-01-01",
            "2024-01-01",
            cache_dir=tmp_path,
            session=session,
        )

    assert len(session.calls) == 1
