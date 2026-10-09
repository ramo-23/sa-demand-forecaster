import time

import numpy as np
import pandas as pd
import pytest

import src.data.synthetic as synthetic
from src.evaluate.run_backtest import (
    RESULT_COLUMNS,
    _scale_intervals,
    _split_training_for_calibration,
    run_backtest,
)

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
        calibration_hours=24,
        calibration_fraction=0.2,
    )
    elapsed = time.perf_counter() - started_at

    assert results.columns.tolist() == RESULT_COLUMNS
    assert set(results["model"]) == {
        "seasonal_naive",
        "lgbm_quantile_full",
        "lgbm_conformal_90d",
        "lgbm_conformal_20pct",
    }
    assert results.loc[results["model"] == "seasonal_naive", "mape"].iloc[0] == 0.0
    lgbm_coverage = results.loc[
        results["model"] == "lgbm_quantile_full",
        ["coverage_80", "coverage_95"],
    ]
    assert lgbm_coverage.ge(0.0).all().all()
    assert lgbm_coverage.le(1.0).all().all()
    assert len(results) == 4
    assert elapsed < 60.0
    fold_boundaries = results.attrs["fold_boundaries"]
    assert len(fold_boundaries) == 1
    for fold in fold_boundaries:
        row_sets = fold["row_sets"]
        for variant in ("lgbm_conformal_90d", "lgbm_conformal_20pct"):
            boundaries = row_sets[variant]
            assert (
                boundaries["fit_end"]
                < boundaries["cal_start"]
                <= boundaries["cal_end"]
                < boundaries["test_start"]
            )


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
        calibration_hours=24,
        calibration_fraction=0.2,
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
    assert segmented["n_points"].sum() == (
        len(results["model"].unique()) * overall["n_points"].iloc[0]
    )
    assert segmented["n_distinct_days"].gt(0).all()


def test_run_backtest_preserves_distinct_day_count_for_each_custom_mlr_bin(
    monkeypatch,
) -> None:
    demand, weather = _periodic_demand_and_weather(monkeypatch, days=60)
    group_labels = pd.Series(
        np.resize(["0", "(0, 1300]", "(1300, 2100]"], len(demand)),
        index=demand.index,
    )
    year_labels = pd.Series(
        np.resize(["2024|mlr>0", "2024|mlr==0"], len(demand)),
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
        groupings={"year": year_labels},
        calibration_hours=24,
        calibration_fraction=0.2,
    )

    grouped = results.loc[
        results["grouping"].isna() & results["segment"].ne("all")
    ]
    assert set(grouped["segment"]) == {"0", "(0, 1300]", "(1300, 2100]"}
    assert grouped["n_distinct_days"].gt(0).all()
    assert grouped.groupby("segment")["n_distinct_days"].nunique().eq(1).all()
    yearly = results.loc[results["grouping"] == "year"]
    assert set(yearly["segment"]) == {"2024|mlr>0", "2024|mlr==0"}
    assert set(yearly["model"]) == {
        "lgbm_quantile_full",
        "seasonal_naive",
        "lgbm_conformal_90d",
        "lgbm_conformal_20pct",
    }


def test_calibration_blocks_are_after_fit_and_before_test() -> None:
    train_index = pd.date_range(
        "2024-01-01",
        periods=1000,
        freq="h",
        tz=TIMEZONE,
    )
    splits = _split_training_for_calibration(
        train_index,
        calibration_hours=216,
        calibration_fraction=0.2,
    )
    test_start = train_index[-1] + pd.Timedelta(hours=1)

    for fit_index, calibration_index in splits.values():
        assert fit_index[-1] < calibration_index[0]
        assert calibration_index[-1] < test_start
    assert len(splits["90d"][1]) == 216
    assert len(splits["20pct"][1]) == 200


def test_run_backtest_records_boundaries_and_adjacent_90d_calibration_for_every_fold(
    monkeypatch,
) -> None:
    demand, weather = _periodic_demand_and_weather(monkeypatch, days=60)
    results = run_backtest(
        demand,
        weather,
        initial_train_hours=700,
        step_hours=120,
        calibration_hours=72,
        calibration_fraction=0.2,
    )

    boundaries = results.attrs["fold_boundaries"]

    assert len(boundaries) == int(results["n_folds"].iloc[0])
    assert len(boundaries) >= 5
    for fold in boundaries:
        assert set(fold["row_sets"]) == {
            "lgbm_quantile_full",
            "lgbm_conformal_90d",
            "lgbm_conformal_20pct",
        }
        for variant in ("lgbm_conformal_90d", "lgbm_conformal_20pct"):
            fold_values = fold["row_sets"][variant]
            assert (
                fold_values["fit_end"]
                < fold_values["cal_start"]
                <= fold_values["cal_end"]
                < fold_values["test_start"]
            )
        ninety_day = fold["row_sets"]["lgbm_conformal_90d"]
        assert ninety_day["test_start"] - ninety_day["cal_end"] == pd.Timedelta(
            hours=1
        )
        full = fold["row_sets"]["lgbm_quantile_full"]
        assert full["fit_end"] + pd.Timedelta(hours=1) == full["test_start"]


def test_run_backtest_rejects_calibration_block_overlapping_fit_rows(
    monkeypatch,
) -> None:
    demand, weather = _periodic_demand_and_weather(monkeypatch, days=60)

    def overlapping_split(train_index, calibration_hours, calibration_fraction, **kwargs):
        eligible = kwargs["finite_train_index"]
        overlap = (eligible, eligible[-72:])
        return {"90d": overlap, "20pct": overlap}

    monkeypatch.setattr(
        "src.evaluate.run_backtest._split_training_for_calibration",
        overlapping_split,
    )

    with pytest.raises(ValueError, match=r"fold 0 lgbm_conformal_90d boundaries invalid"):
        run_backtest(
            demand,
            weather,
            initial_train_hours=700,
            step_hours=120,
            calibration_hours=72,
            calibration_fraction=0.2,
        )


def test_positive_scaling_preserves_quantile_order() -> None:
    intervals = pd.DataFrame(
        {
            "lower_95": [0.1, 0.2],
            "lower_80": [0.3, 0.4],
            "median": [0.5, 0.6],
            "upper_80": [0.7, 0.8],
            "upper_95": [0.9, 1.0],
        }
    )

    scaled = _scale_intervals(intervals, np.array([10.0, 100.0]))

    assert (scaled["lower_95"] <= scaled["lower_80"]).all()
    assert (scaled["lower_80"] <= scaled["median"]).all()
    assert (scaled["median"] <= scaled["upper_80"]).all()
    assert (scaled["upper_80"] <= scaled["upper_95"]).all()


def test_ratio_168_beats_level_after_multiplicative_test_level_shift(
    monkeypatch,
) -> None:
    demand, weather = _periodic_demand_and_weather(monkeypatch, days=60)
    shift_start = 30 * 24
    demand.loc[demand.index[shift_start:], "demand_mw"] *= 1.5

    results = run_backtest(
        demand,
        weather,
        initial_train_hours=shift_start,
        step_hours=480,
        horizon_hours=480,
        calibration_hours=168,
        calibration_fraction=0.2,
        target="ratio_168",
    )

    metrics = results.set_index("model")["mape"]
    assert metrics["lgbm_quantile_full_ratio"] < metrics["lgbm_quantile_full"]
    assert metrics["lgbm_conformal_90d_ratio"] < metrics["lgbm_conformal_90d"]
    predictions = results.attrs["predictions"]
    assert len(predictions) == 480
    assert {
        "timestamp",
        "fold",
        "y",
        "lgbm_quantile_full_ratio_point",
        "lgbm_quantile_full_ratio_lower_80",
        "lgbm_quantile_full_ratio_upper_80",
        "lgbm_quantile_full_ratio_lower_95",
        "lgbm_quantile_full_ratio_upper_95",
    }.issubset(predictions.columns)
    for fold in results.attrs["fold_boundaries"]:
        row_sets = fold["row_sets"]
        assert row_sets["lgbm_quantile_full_ratio"]["fit_end"] == row_sets[
            "lgbm_quantile_full"
        ]["fit_end"]
        assert row_sets["lgbm_quantile_full_ratio"]["test_start"] == row_sets[
            "lgbm_quantile_full"
        ]["test_start"]
        assert row_sets["lgbm_conformal_90d_ratio"] == {
            **row_sets["lgbm_conformal_90d"],
            "row_set": "lgbm_conformal_90d_ratio",
        }


def test_ratio_168_rejects_horizons_above_168(monkeypatch) -> None:
    demand, weather = _periodic_demand_and_weather(monkeypatch)

    with pytest.raises(ValueError, match="horizon must be <= 168"):
        run_backtest(
            demand,
            weather,
            horizon=169,
            initial_train_hours=600,
            step_hours=24,
            target="ratio_168",
            calibration_hours=24,
        )
