import argparse
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.loader import TIMEZONE, load_demand, load_eskom
from src.evaluate.metrics import mape
from src.evaluate.run_backtest import print_fold_boundaries, run_backtest
from src.evaluate.run_backtest import TARGET_CHOICES
from src.features.weather import fetch_weather

DEMAND_PATH = Path("data/raw/ESK19907.csv")
RESULT_PATH = Path("results/backtest_summary.csv")
TRAINING_START = pd.Timestamp("2022-04-01", tz=TIMEZONE)
DEFAULT_TEST_START = "2023-04-01"
FORECAST_HORIZON_HOURS = 24
MONTHLY_FOLD_HOURS = 720
MLR_BIN_LABELS = (
    "0",
    "(0, 1300]",
    "(1300, 2100]",
    "(2100, 2900]",
    ">2900",
)
SHED_FLAG_LABELS = ("mlr>0", "mlr==0")
MINIMUM_REPORTED_POINTS = 100
SUMMARY_METRICS = (
    "mape",
    "mase",
    "coverage_80",
    "coverage_95",
    "n_points",
    "n_distinct_days",
)
PAIRED_MONTH_MINIMUM_HOURS = 48
PAIRED_MONTH_COLUMNS = [
    "year_month",
    "n_shed",
    "n_nonshed",
    "mape_shed",
    "mape_nonshed",
    "diff",
    "cov80_shed",
    "cov80_nonshed",
]


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run real Eskom demand backtest.")
    parser.add_argument(
        "--test-start",
        default=DEFAULT_TEST_START,
        help=f"First test date, inclusive (default: {DEFAULT_TEST_START})",
    )
    parser.add_argument(
        "--calibration-hours",
        type=int,
        default=2160,
        help="Calibration observations for lgbm_conformal_90d (default: 2160)",
    )
    parser.add_argument(
        "--calibration-fraction",
        type=float,
        default=0.2,
        help="Calibration fraction for lgbm_conformal_20pct (default: 0.2)",
    )
    parser.add_argument(
        "--target",
        choices=TARGET_CHOICES,
        default="level",
        help="Forecast target representation (default: level)",
    )
    return parser.parse_args(argv)


def _mlr_bins(mlr: pd.Series) -> pd.Series:
    if mlr.isna().any():
        timestamps = ", ".join(
            str(timestamp) for timestamp in mlr.index[mlr.isna()]
        )
        raise ValueError(f"MLR values are missing at timestamps: {timestamps}")
    values = pd.to_numeric(mlr, errors="raise")
    if (values < 0).any():
        raise ValueError("MLR values must be non-negative")
    bins = np.select(
        [
            values.eq(0),
            values.gt(0) & values.le(1300),
            values.gt(1300) & values.le(2100),
            values.gt(2100) & values.le(2900),
            values.gt(2900),
        ],
        MLR_BIN_LABELS,
        default="",
    )
    if (bins == "").any():
        raise ValueError("Could not assign every MLR value to a configured bin")
    return pd.Series(bins, index=mlr.index, name="mlr_bin")


def _build_summary_groupings(
    index: pd.DatetimeIndex,
    mlr: pd.Series,
) -> dict[str, pd.Series]:
    if not isinstance(index, pd.DatetimeIndex) or index.tz is None:
        raise TypeError("summary grouping requires a timezone-aware DatetimeIndex")
    if not isinstance(mlr, pd.Series) or not mlr.index.equals(index):
        raise ValueError("MLR index must exactly match the demand index")
    if mlr.isna().any():
        timestamps = ", ".join(str(timestamp) for timestamp in mlr.index[mlr.isna()])
        raise ValueError(f"MLR values are missing at timestamps: {timestamps}")
    values = pd.to_numeric(mlr, errors="raise")
    if (values < 0).any():
        raise ValueError("MLR values must be non-negative")

    shed_flag = np.where(values.to_numpy() > 0, "mlr>0", "mlr==0")
    year_labels = pd.Series(
        [f"{year}|{flag}" for year, flag in zip(index.year, shed_flag)],
        index=index,
        name="year_shed_flag",
    )
    daypart_names = np.select(
        [
            index.hour <= 5,
            (index.hour >= 6) & (index.hour <= 11),
            (index.hour >= 12) & (index.hour <= 17),
            index.hour >= 18,
        ],
        ["00-05", "06-11", "12-17", "18-23"],
        default="",
    )
    daypart_labels = pd.Series(
        [f"{daypart}|{flag}" for daypart, flag in zip(daypart_names, shed_flag)],
        index=index,
        name="daypart_shed_flag",
    )
    month_labels = pd.Series(
        [
            f"{year:04d}-{month:02d}|{flag}"
            for year, month, flag in zip(index.year, index.month, shed_flag)
        ],
        index=index,
        name="month_shed_flag",
    )
    return {
        "year": year_labels,
        "daypart": daypart_labels,
        "month_shed": month_labels,
    }


def _prepare_mlr_slices(mlr: pd.Series, test_start: pd.Timestamp) -> pd.Series:
    test_mlr = mlr.loc[mlr.index >= test_start]
    _mlr_bins(test_mlr)
    labels = pd.Series(0.0, index=mlr.index, name=mlr.name)
    labels.loc[test_mlr.index] = test_mlr
    return labels


def _summary_table(
    results: pd.DataFrame,
    grouping: str,
    dimension: str,
) -> pd.DataFrame:
    columns = [dimension, "shed_flag", "model", *SUMMARY_METRICS]
    selected = results.loc[
        results["grouping"].eq(grouping)
        & results["model"].isin(
            (
                "lgbm_quantile_full",
                "lgbm_conformal_90d",
                "lgbm_conformal_20pct",
                "lgbm_quantile_full_ratio",
                "lgbm_conformal_90d_ratio",
                "seasonal_naive",
            )
        )
    ]
    rows: list[dict[str, object]] = []
    for row in selected.to_dict(orient="records"):
        dimension_value, _, shed_flag = str(row["segment"]).partition("|")
        if not shed_flag:
            raise ValueError(
                f"invalid {grouping} grouping label: {row['segment']!r}"
            )
        value: object = (
            int(dimension_value) if dimension == "year" else dimension_value
        )
        summary: dict[str, object] = {
            dimension: value,
            "shed_flag": shed_flag,
            "model": row["model"],
            **{metric: row.get(metric) for metric in SUMMARY_METRICS},
        }
        if int(row["n_points"]) < MINIMUM_REPORTED_POINTS:
            for metric in SUMMARY_METRICS:
                summary[metric] = "suppressed"
        rows.append(summary)
    return pd.DataFrame(rows, columns=columns)


def _paired_month_table(results: pd.DataFrame) -> pd.DataFrame:
    selected = results.loc[
        results["grouping"].eq("month_shed")
        & results["model"].eq("lgbm_quantile_full")
    ]
    rows: list[dict[str, object]] = []
    month_labels = sorted(
        {
            str(segment).partition("|")[0]
            for segment in selected["segment"].dropna()
        }
    )
    for year_month in month_labels:
        shed = selected.loc[selected["segment"] == f"{year_month}|mlr>0"]
        nonshed = selected.loc[selected["segment"] == f"{year_month}|mlr==0"]
        if shed.empty or nonshed.empty:
            continue
        shed_row = shed.iloc[0]
        nonshed_row = nonshed.iloc[0]
        n_shed = int(shed_row["n_points"])
        n_nonshed = int(nonshed_row["n_points"])
        if (
            n_shed < PAIRED_MONTH_MINIMUM_HOURS
            or n_nonshed < PAIRED_MONTH_MINIMUM_HOURS
        ):
            continue
        mape_shed = float(shed_row["mape"])
        mape_nonshed = float(nonshed_row["mape"])
        rows.append(
            {
                "year_month": year_month,
                "n_shed": n_shed,
                "n_nonshed": n_nonshed,
                "mape_shed": mape_shed,
                "mape_nonshed": mape_nonshed,
                "diff": mape_shed - mape_nonshed,
                "cov80_shed": float(shed_row["coverage_80"]),
                "cov80_nonshed": float(nonshed_row["coverage_80"]),
            }
        )
    return pd.DataFrame(rows, columns=PAIRED_MONTH_COLUMNS)


def _forecast_benchmark_row(
    demand: pd.Series,
    forecast: pd.Series,
    test_index: pd.DatetimeIndex,
) -> dict[str, object]:
    actual = demand.reindex(test_index)
    predicted = forecast.reindex(test_index)
    if actual.isna().any():
        raise ValueError("demand is missing on forecast benchmark test hours")
    if predicted.isna().any():
        timestamps = ", ".join(
            str(timestamp) for timestamp in predicted.index[predicted.isna()]
        )
        raise ValueError(
            "RSA contracted forecast is missing on test timestamps: "
            f"{timestamps}"
        )
    row: dict[str, object] = {
        "segment": "all",
        "n_distinct_days": np.nan,
        "model": "eskom_rsa_contracted_forecast",
        "mape": mape(actual.to_numpy(dtype=float), predicted.to_numpy(dtype=float)),
        "mase": np.nan,
        "coverage_80": np.nan,
        "coverage_95": np.nan,
        "width_80": np.nan,
        "width_95": np.nan,
        "n_folds": np.nan,
        "n_points": np.nan,
        "issue_time_status": "issue time unverified",
    }
    return row


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    try:
        test_start_date = pd.Timestamp(args.test_start).date()
    except (TypeError, ValueError) as error:
        raise ValueError("--test-start must be a valid YYYY-MM-DD date") from error
    if test_start_date.isoformat() != args.test_start:
        raise ValueError("--test-start must use YYYY-MM-DD format")
    test_start = pd.Timestamp(test_start_date, tz=TIMEZONE)

    demand = load_demand(DEMAND_PATH)
    demand = demand.loc[demand.index >= TRAINING_START]
    if demand.empty or demand.index[0] != TRAINING_START:
        raise ValueError(
            "Demand data must include the hourly timestamp "
            f"{TRAINING_START} to start the requested training span"
        )
    if test_start not in demand.index:
        raise ValueError(
            f"Demand data must include the test start timestamp {test_start}"
        )

    eskom = load_eskom(DEMAND_PATH)
    mlr = eskom["mlr"].reindex(demand.index)
    rsa_forecast = eskom["rsa_contracted_forecast"].reindex(demand.index)
    mlr_for_slicing = _prepare_mlr_slices(mlr, test_start)
    mlr_groups = _mlr_bins(mlr_for_slicing)
    groupings = _build_summary_groupings(demand.index, mlr_for_slicing)
    weather = fetch_weather(
        demand.index[0].date().isoformat(),
        demand.index[-1].date().isoformat(),
    )

    initial_train_hours = int(demand.index.get_loc(test_start))
    results = run_backtest(
        demand,
        weather,
        horizon=FORECAST_HORIZON_HOURS,
        initial_train_hours=initial_train_hours,
        step_hours=MONTHLY_FOLD_HOURS,
        season=168,
        horizon_hours=MONTHLY_FOLD_HOURS,
        include_partial_test=True,
        mlr_groups=mlr_groups,
        groupings=groupings,
        calibration_hours=args.calibration_hours,
        calibration_fraction=args.calibration_fraction,
        target=args.target,
    )
    print_fold_boundaries(results.attrs["fold_boundaries"])

    test_index = demand.index[initial_train_hours:]
    lgbm_overall = results.loc[
        results["segment"].eq("all")
        & results["model"].eq("lgbm_quantile_full")
    ]
    if lgbm_overall.empty or int(lgbm_overall["n_points"].iloc[0]) != len(test_index):
        raise ValueError(
            "backtest test hours do not match the contiguous span from --test-start"
        )
    benchmark_row = _forecast_benchmark_row(
        demand["demand_mw"],
        rsa_forecast,
        test_index,
    )
    prediction_output = results.attrs["predictions"]
    prediction_output["eskom_rsa_contracted_forecast_point"] = (
        rsa_forecast.reindex(pd.DatetimeIndex(prediction_output["timestamp"]))
        .to_numpy(dtype=float)
    )
    for bound in ("lower_80", "upper_80", "lower_95", "upper_95"):
        prediction_output[f"eskom_rsa_contracted_forecast_{bound}"] = np.nan
    results = pd.concat(
        [results, pd.DataFrame([benchmark_row])],
        ignore_index=True,
        sort=False,
    )

    overall = results.loc[results["segment"] == "all"]
    by_mlr = results.loc[results["segment"].isin(MLR_BIN_LABELS)]
    by_year = _summary_table(results, "year", "year")
    by_daypart = _summary_table(results, "daypart", "daypart")
    paired_months = _paired_month_table(results)
    negative_month_count = int(paired_months["diff"].lt(0).sum())
    median_month_diff = (
        float(paired_months["diff"].median())
        if not paired_months.empty
        else float("nan")
    )
    test_points = int(overall["n_points"].max())
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    prediction_path = Path("results") / f"backtest_predictions_{args.target}.csv"
    prediction_output.to_csv(prediction_path, index=False)
    csv_results = pd.concat(
        [
            results.assign(table="overall_and_mlr_bins"),
            by_year.assign(table="year_x_shed_flag"),
            by_daypart.assign(table="daypart_x_shed_flag"),
            paired_months.assign(table="paired_month_shed_vs_nonshed"),
        ],
        ignore_index=True,
        sort=False,
    )
    csv_results.to_csv(RESULT_PATH, index=False)

    first_test_timestamp = test_start
    last_test_timestamp = demand.index[-1]
    print(
        f"REAL DATA BACKTEST: {first_test_timestamp} through "
        f"{last_test_timestamp}; {test_points} test points"
    )
    print(f"MLR bins: {', '.join(MLR_BIN_LABELS)} MW")
    print("MLR bin edges are arbitrary, roughly quartiles of positive MLR hours.")
    print(f"Results written to {RESULT_PATH}")
    print(f"Per-prediction results written to {prediction_path}")
    print(overall.to_string(index=False))
    print("\nError, coverage, and distinct days by MLR bin:")
    print(by_mlr.to_string(index=False))
    print("\nYear x shed_flag:")
    print(by_year.to_string(index=False))
    print("\nDaypart x shed_flag (local hour):")
    print(by_daypart.to_string(index=False))
    print("\nPaired monthly shed vs non-shed, lgbm_quantile_full:")
    print("The 48-hour minimum per group is arbitrary.")
    print(paired_months.to_string(index=False))
    print(
        f"Months with shed-minus-non-shed MAPE difference < 0: "
        f"{negative_month_count}"
    )
    print(f"Median monthly MAPE difference: {median_month_diff}")
    print(
        "\nConformal time-series caveat: split-conformal coverage relies on "
        "exchangeability, which is generally not guaranteed for serially "
        "dependent demand."
    )


if __name__ == "__main__":
    main()
