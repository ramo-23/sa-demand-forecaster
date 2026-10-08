import time

import numpy as np
import pandas as pd

import src.data.synthetic as synthetic
from src.evaluate.run_backtest import RESULT_COLUMNS, run_backtest

TIMEZONE = "Africa/Johannesburg"


def _periodic_demand_and_weather(
    monkeypatch,
    days: int = 30,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    def fake_fetch_weather(start_date: str, end_date: str) -> pd.DataFrame:
        index = pd.date_range(
            start=start_date,
            end=f"{end_date} 23:00",
            freq="h",
            tz=TIMEZONE,
            name="timestamp",
        )
        hours = np.arange(len(index), dtype=float)
        return pd.DataFrame(
            {
                "temp_weighted": 20.0 + np.sin(hours / 100),
                "temp_johannesburg": 21.0 + np.sin(hours / 100),
                "temp_cape_town": 18.0 + np.sin(hours / 100),
                "temp_durban": 23.0 + np.sin(hours / 100),
                "temp_bloemfontein": 17.0 + np.sin(hours / 100),
            },
            index=index,
        )

    monkeypatch.setattr(synthetic, "fetch_weather", fake_fetch_weather)
    start = "2024-01-01"
    end_date = pd.Timestamp(start) + pd.Timedelta(days=days - 1)
    end = end_date.date().isoformat()
    index = pd.date_range(
        start,
        end=f"{end} 23:00",
        freq="h",
        tz=TIMEZONE,
        name="timestamp",
    )
    season = np.array([30000.0 + i * 10 for i in range(168)])
    demand_values = np.tile(season, (len(index) + len(season) - 1) // len(season))
    demand = pd.DataFrame({"demand_mw": demand_values[: len(index)]}, index=index)
    weather = fake_fetch_weather(start, end)
    return demand, weather


def test_run_backtest_returns_expected_results_on_30_days_under_one_minute(
    monkeypatch,
) -> None:
    demand, weather = _periodic_demand_and_weather(monkeypatch)

    started_at = time.perf_counter()
    results = run_backtest(
        demand,
        weather,
        horizon=24,
        initial_train_hours=600,
        step_hours=120,
        season=168,
    )
    elapsed = time.perf_counter() - started_at

    assert results.columns.tolist() == RESULT_COLUMNS
    assert results["model"].tolist() == ["seasonal_naive", "lgbm_quantile"]
    assert results.loc[results["model"] == "seasonal_naive", "mape"].iloc[0] == 0.0
    lgbm_coverage = results.loc[
        results["model"] == "lgbm_quantile",
        ["coverage_80", "coverage_95"],
    ]
    assert lgbm_coverage.ge(0.0).all().all()
    assert lgbm_coverage.le(1.0).all().all()
    assert len(results) == 2
    assert elapsed < 60.0


def test_run_backtest_separates_forecast_horizon_from_monthly_fold_and_groups_mlr(
    monkeypatch,
) -> None:
    demand, weather = _periodic_demand_and_weather(monkeypatch, days=60)
    mlr = pd.Series(
        np.tile([0.0, 10.0], len(demand) // 2),
        index=demand.index,
    )

    results = run_backtest(
        demand,
        weather,
        horizon=24,
        initial_train_hours=600,
        step_hours=720,
        horizon_hours=720,
        include_partial_test=True,
        mlr=mlr,
    )

    assert set(results["segment"].drop_duplicates()) == {
        "all",
        "mlr>0",
        "mlr==0",
    }
    overall = results.loc[results["segment"] == "all"]
    assert overall["n_folds"].eq(2).all()
    assert overall["n_points"].eq(840).all()
    naive = results.loc[results["model"] == "seasonal_naive"]
    assert naive["mape"].eq(0.0).all()
    segmented = results.loc[results["segment"] != "all"]
    assert segmented["n_points"].sum() == 2 * overall["n_points"].iloc[0]
    assert segmented["n_distinct_days"].gt(0).all()


def test_run_backtest_preserves_distinct_day_count_for_each_custom_mlr_bin(
    monkeypatch,
) -> None:
    demand, weather = _periodic_demand_and_weather(monkeypatch, days=60)
    group_labels = pd.Series(
        np.resize(["0", "(0, 1300]", "(1300, 2100]"], len(demand)),
        index=demand.index,
    )

    results = run_backtest(
        demand,
        weather,
        initial_train_hours=600,
        step_hours=720,
        horizon_hours=720,
        include_partial_test=True,
        mlr_groups=group_labels,
    )

    grouped = results.loc[results["segment"] != "all"]
    assert set(grouped["segment"]) == {"0", "(0, 1300]", "(1300, 2100]"}
    assert grouped["n_distinct_days"].gt(0).all()
    assert grouped.groupby("segment")["n_distinct_days"].nunique().eq(1).all()
