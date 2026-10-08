import argparse
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.loader import TIMEZONE, load_demand, load_eskom
from src.evaluate.run_backtest import run_backtest
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


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run real Eskom demand backtest.")
    parser.add_argument(
        "--test-start",
        default=DEFAULT_TEST_START,
        help=f"First test date, inclusive (default: {DEFAULT_TEST_START})",
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
    mlr_groups = _mlr_bins(mlr)
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
    )

    overall = results.loc[results["segment"] == "all"]
    by_mlr = results.loc[results["segment"] != "all"]
    test_points = int(overall["n_points"].max())
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(RESULT_PATH, index=False)

    first_test_timestamp = test_start
    last_test_timestamp = demand.index[-1]
    print(
        f"REAL DATA BACKTEST: {first_test_timestamp} through "
        f"{last_test_timestamp}; {test_points} test points"
    )
    print(f"MLR bins: {', '.join(MLR_BIN_LABELS)} MW")
    print("MLR bin edges are arbitrary, roughly quartiles of positive MLR hours.")
    print(f"Overall and binned results written to {RESULT_PATH}")
    print(overall.to_string(index=False))
    print("\nError, coverage, and distinct days by MLR bin:")
    print(by_mlr.to_string(index=False))


if __name__ == "__main__":
    main()
