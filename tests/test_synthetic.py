import numpy as np
import pandas as pd

import src.data.synthetic as synthetic

TIMEZONE = "Africa/Johannesburg"


def test_synthetic_demand_is_reproducible_and_matches_contract(monkeypatch) -> None:
    def fake_fetch_weather(start_date: str, end_date: str) -> pd.DataFrame:
        index = pd.date_range(
            start=start_date,
            end=f"{end_date} 23:00",
            freq="h",
            tz=TIMEZONE,
            name="timestamp",
        )
        return pd.DataFrame({"temp_weighted": np.full(len(index), 20.0)}, index=index)

    monkeypatch.setattr(synthetic, "fetch_weather", fake_fetch_weather)

    first = synthetic.make_synthetic_demand("2024-01-01", "2024-01-02", seed=7)
    second = synthetic.make_synthetic_demand("2024-01-01", "2024-01-02", seed=7)

    assert first.columns.tolist() == ["demand_mw"]
    assert first.index.name == "timestamp"
    assert str(pd.DatetimeIndex(first.index).tz) == TIMEZONE
    assert first.index.is_monotonic_increasing
    assert len(first) == 48
    pd.testing.assert_frame_equal(first, second)
