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
        start = date.fromisoformat(params["start_date"])
        end = date.fromisoformat(params["end_date"])
        times = pd.date_range(
            start,
            f"{end.isoformat()} 23:00",
            freq="h",
        )
        response = requests.Response()
        response.status_code = self.status_code
        response._content = b"service unavailable" if self.status_code != 200 else b""
        if self.status_code != 200:
            return response
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


def test_2026_request_chunk_ends_on_requested_date(tmp_path: Path) -> None:
    session = FakeSession()
    requested_end = "2026-03-15"

    fetch_weather(
        "2026-03-10",
        requested_end,
        cache_dir=tmp_path,
        session=session,
    )

    assert len(session.calls) == len(CITIES)
    assert all(params["end_date"] == requested_end for _, params in session.calls)


def test_partial_year_cache_is_not_reused_for_later_requested_end_date(
    tmp_path: Path,
) -> None:
    session = FakeSession()

    fetch_weather("2026-02-01", "2026-02-02", tmp_path, session)
    assert len(session.calls) == len(CITIES)

    fetch_weather("2026-02-01", "2026-02-03", tmp_path, session)

    assert len(session.calls) == len(CITIES) * 2
    assert all(params["end_date"] == "2026-02-03" for _, params in session.calls[-4:])


def test_non_200_response_raises_and_does_not_return_partial_data(
    tmp_path: Path,
) -> None:
    session = FakeSession(status_code=503)

    with pytest.raises(
        requests.HTTPError,
        match="HTTP 503.*Response body: service unavailable",
    ):
        fetch_weather(
            "2024-01-01",
            "2024-01-01",
            cache_dir=tmp_path,
            session=session,
        )

    assert len(session.calls) == 1


def test_missing_temperatures_report_first_and_last_timestamp(
    tmp_path: Path,
) -> None:
    class MissingTemperatureSession(FakeSession):
        def get(self, url: str, **kwargs: object) -> requests.Response:
            response = super().get(url, **kwargs)
            if response.status_code == 200:
                payload = response.json()
                payload["hourly"]["temperature_2m"][0] = None
                payload["hourly"]["temperature_2m"][-1] = None
                response._content = json.dumps(payload).encode("utf-8")
            return response

    with pytest.raises(
        ValueError,
        match="2026-02-01 00:00:00\\+02:00 through 2026-02-01 23:00:00\\+02:00",
    ):
        fetch_weather(
            "2026-02-01",
            "2026-02-01",
            cache_dir=tmp_path,
            session=MissingTemperatureSession(),
        )
