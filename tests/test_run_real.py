import numpy as np
import pandas as pd
import pytest

from src.evaluate.run_real import (
    DEFAULT_TEST_START,
    MLR_BIN_LABELS,
    MINIMUM_REPORTED_POINTS,
    PAIRED_MONTH_COLUMNS,
    PAIRED_MONTH_MINIMUM_HOURS,
    SUMMARY_METRICS,
    _build_summary_groupings,
    _mlr_bins,
    _parse_args,
    _paired_month_table,
    _prepare_mlr_slices,
    _forecast_benchmark_row,
    _summary_table,
)


def test_mlr_bins_use_requested_inclusive_upper_edges() -> None:
    values = [0, 0.1, 1300, 1300.1, 2100, 2100.1, 2900, 2900.1]
    mlr = pd.Series(values, index=pd.RangeIndex(len(values)))

    bins = _mlr_bins(mlr)

    assert bins.tolist() == [
        "0",
        "(0, 1300]",
        "(0, 1300]",
        "(1300, 2100]",
        "(1300, 2100]",
        "(2100, 2900]",
        "(2100, 2900]",
        ">2900",
    ]
    assert MLR_BIN_LABELS == (
        "0",
        "(0, 1300]",
        "(1300, 2100]",
        "(2100, 2900]",
        ">2900",
    )


def test_mlr_bins_reject_missing_and_negative_values() -> None:
    with pytest.raises(ValueError, match="MLR values are missing"):
        _mlr_bins(pd.Series([0.0, np.nan]))
    with pytest.raises(ValueError, match="non-negative"):
        _mlr_bins(pd.Series([0.0, -1.0]))


def test_test_start_cli_defaults_to_2023_april_first() -> None:
    args = _parse_args([])
    assert args.test_start == DEFAULT_TEST_START == "2023-04-01"
    assert args.calibration_hours == 2160
    assert args.calibration_fraction == 0.2
    assert args.target == "level"
    assert _parse_args(["--test-start", "2024-10-01"]).test_start == "2024-10-01"
    assert _parse_args(["--target", "ratio_168"]).target == "ratio_168"


def test_summary_groupings_use_local_year_daypart_and_mlr_flag() -> None:
    index = pd.date_range(
        "2023-12-31 00:00",
        "2024-01-01 23:00",
        freq="h",
        tz="Africa/Johannesburg",
        name="timestamp",
    )
    mlr = pd.Series(np.resize([0.0, 1.0], len(index)), index=index)

    groupings = _build_summary_groupings(index, mlr)

    assert groupings["year"].loc[index[0]] == "2023|mlr==0"
    assert groupings["year"].loc[index[-1]] == "2024|mlr>0"
    assert groupings["daypart"].loc[index[5]] == "00-05|mlr>0"
    assert groupings["daypart"].loc[index[6]] == "06-11|mlr==0"
    assert groupings["daypart"].loc[index[12]] == "12-17|mlr==0"
    assert groupings["daypart"].loc[index[18]] == "18-23|mlr==0"
    assert groupings["month_shed"].loc[index[0]] == "2023-12|mlr==0"
    assert groupings["month_shed"].loc[index[-1]] == "2024-01|mlr>0"


def test_mlr_slice_preparation_ignores_pretest_values_but_validates_test_values() -> None:
    index = pd.date_range(
        "2024-01-01",
        periods=3,
        freq="h",
        tz="Africa/Johannesburg",
    )
    mlr = pd.Series([-1.0, 0.0, 1.0], index=index)

    prepared = _prepare_mlr_slices(mlr, index[1])

    assert prepared.tolist() == [0.0, 0.0, 1.0]
    with pytest.raises(ValueError, match="MLR values are missing"):
        _prepare_mlr_slices(mlr.mask(mlr.index == index[2]), index[1])


def test_summary_table_suppresses_small_cells_and_keeps_only_requested_models() -> None:
    result_rows = []
    for grouping, segment, n_points in (
        ("year", "2024|mlr>0", MINIMUM_REPORTED_POINTS - 1),
        ("daypart", "06-11|mlr==0", MINIMUM_REPORTED_POINTS),
    ):
        for model in (
            "lgbm_quantile_full",
            "lgbm_conformal_90d",
            "lgbm_conformal_20pct",
            "seasonal_naive",
            "unrelated_model",
        ):
            result_rows.append(
                {
                    "grouping": grouping,
                    "segment": segment,
                    "model": model,
                    "mape": 0.25,
                    "mase": 0.5,
                    "coverage_80": 0.8,
                    "coverage_95": 0.95,
                    "n_points": n_points,
                    "n_distinct_days": 10,
                }
            )
    results = pd.DataFrame(result_rows)

    year_table = _summary_table(results, "year", "year")
    daypart_table = _summary_table(results, "daypart", "daypart")

    assert year_table["model"].tolist() == [
        "lgbm_quantile_full",
        "lgbm_conformal_90d",
        "lgbm_conformal_20pct",
        "seasonal_naive",
    ]
    assert year_table.loc[:, SUMMARY_METRICS].eq("suppressed").all().all()
    assert daypart_table["model"].tolist() == [
        "lgbm_quantile_full",
        "lgbm_conformal_90d",
        "lgbm_conformal_20pct",
        "seasonal_naive",
    ]
    assert daypart_table["n_points"].tolist() == [MINIMUM_REPORTED_POINTS] * 4
    assert daypart_table["mape"].tolist() == [0.25] * 4


def test_paired_month_table_requires_48_hours_for_each_shed_group() -> None:
    results = pd.DataFrame(
        [
            {
                "grouping": "month_shed",
                "segment": f"{month}|{flag}",
                "model": "lgbm_quantile_full",
                "n_points": count,
                "mape": error,
                "coverage_80": coverage,
            }
            for month, flag, count, error, coverage in (
                ("2024-01", "mlr>0", 48, 0.10, 0.80),
                ("2024-01", "mlr==0", 48, 0.15, 0.90),
                ("2024-02", "mlr>0", 47, 0.20, 0.70),
                ("2024-02", "mlr==0", 60, 0.10, 0.85),
            )
        ]
    )

    table = _paired_month_table(results)

    assert table.columns.tolist() == PAIRED_MONTH_COLUMNS
    assert PAIRED_MONTH_MINIMUM_HOURS == 48
    assert table.to_dict(orient="records") == [
        {
            "year_month": "2024-01",
            "n_shed": 48,
            "n_nonshed": 48,
            "mape_shed": 0.10,
            "mape_nonshed": 0.15,
            "diff": pytest.approx(-0.05),
            "cov80_shed": 0.80,
            "cov80_nonshed": 0.90,
        }
    ]


def test_forecast_benchmark_row_scores_only_test_hours_and_marks_issue_time() -> None:
    index = pd.date_range(
        "2024-01-01",
        periods=4,
        freq="h",
        tz="Africa/Johannesburg",
        name="timestamp",
    )
    demand = pd.Series([100.0, 110.0, 120.0, 130.0], index=index)
    forecast = pd.Series([0.0, 100.0, 100.0, 100.0], index=index)
    test_index = index[1:]

    row = _forecast_benchmark_row(demand, forecast, test_index)

    assert row["model"] == "eskom_rsa_contracted_forecast"
    assert row["mape"] == pytest.approx((10 / 110 + 20 / 120 + 30 / 130) / 3)
    assert row["issue_time_status"] == "issue time unverified"
    assert all(
        row[column] != row[column]
        for column in (
            "mase",
            "coverage_80",
            "coverage_95",
            "width_80",
            "width_95",
            "n_folds",
            "n_points",
            "n_distinct_days",
        )
    )
    with pytest.raises(ValueError, match="forecast is missing"):
        _forecast_benchmark_row(demand, forecast.iloc[:2], test_index)
