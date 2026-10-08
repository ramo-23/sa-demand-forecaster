from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.synthetic import make_synthetic_demand
from src.evaluate.backtest import rolling_origin_splits
from src.evaluate.metrics import (
    interval_coverage,
    mape,
    mase,
    mean_interval_width,
)
from src.features.build import build_features
from src.models.lgbm import fit_quantile_models, predict_intervals

RESULT_COLUMNS = [
    "model",
    "mape",
    "mase",
    "coverage_80",
    "coverage_95",
    "width_80",
    "width_95",
    "n_folds",
    "n_points",
]
SEGMENT_COLUMN = "segment"


def _mase_with_scale(
    actual: np.ndarray,
    predicted: np.ndarray,
    first_fold_training: np.ndarray,
    season: int,
) -> float:
    scale = float(
        np.mean(
            np.abs(
                first_fold_training[season:] - first_fold_training[:-season]
            )
        )
    )
    if scale == 0 and np.any(actual != predicted):
        return float("nan")
    return mase(actual, predicted, first_fold_training, season=season)


def _metric_row(
    model_name: str,
    frame: pd.DataFrame,
    first_fold_training: np.ndarray,
    season: int,
    n_folds: int,
) -> dict[str, object]:
    actual = frame["actual"].to_numpy(dtype=float)
    if model_name == "seasonal_naive":
        predicted = frame["seasonal_naive"].to_numpy(dtype=float)
        coverage_80 = coverage_95 = width_80 = width_95 = float("nan")
    else:
        predicted = frame["median"].to_numpy(dtype=float)
        coverage_80 = interval_coverage(
            actual,
            frame["lower_80"].to_numpy(dtype=float),
            frame["upper_80"].to_numpy(dtype=float),
        )
        coverage_95 = interval_coverage(
            actual,
            frame["lower_95"].to_numpy(dtype=float),
            frame["upper_95"].to_numpy(dtype=float),
        )
        width_80 = mean_interval_width(
            frame["lower_80"].to_numpy(dtype=float),
            frame["upper_80"].to_numpy(dtype=float),
        )
        width_95 = mean_interval_width(
            frame["lower_95"].to_numpy(dtype=float),
            frame["upper_95"].to_numpy(dtype=float),
        )

    return {
        "model": model_name,
        "mape": mape(actual, predicted),
        "mase": _mase_with_scale(
            actual,
            predicted,
            first_fold_training,
            season,
        ),
        "coverage_80": coverage_80,
        "coverage_95": coverage_95,
        "width_80": width_80,
        "width_95": width_95,
        "n_folds": n_folds,
        "n_points": len(frame),
    }


def run_backtest(
    demand_df: pd.DataFrame,
    weather_df: pd.DataFrame,
    horizon: int = 24,
    *,
    initial_train_hours: int,
    step_hours: int,
    season: int = 168,
    horizon_hours: int | None = None,
    include_partial_test: bool = False,
    mlr: pd.Series | None = None,
    mlr_groups: pd.Series | None = None,
) -> pd.DataFrame:
    """Run pooled rolling-origin evaluation for seasonal-naive and LightGBM.

    ``horizon`` is the feature/model forecast horizon. ``horizon_hours`` sets
    the test-window length and defaults to ``horizon``. The MASE denominator
    is computed from the raw in-sample demand series of the first fold only.
    Each quantile model is fitted only on finite-feature rows before that
    fold's test-window start.

    When ``mlr`` or ``mlr_groups`` is supplied, results include an overall
    segment and metrics for the requested test-row segments. MLR is used only
    to label and slice already-generated test predictions and is never used as
    a model feature.
    """
    if "demand_mw" not in demand_df.columns:
        raise ValueError("demand_df must contain a 'demand_mw' column")
    if not isinstance(demand_df.index, pd.DatetimeIndex):
        raise TypeError("demand_df must have a pandas DatetimeIndex")
    if not demand_df.columns.is_unique:
        raise ValueError("demand_df columns must be unique")
    if demand_df["demand_mw"].isna().any():
        raise ValueError("demand_mw must not contain missing values")
    if not isinstance(season, int) or season <= 0:
        raise ValueError("season must be a positive integer")
    test_window_hours = horizon if horizon_hours is None else horizon_hours
    if not isinstance(test_window_hours, int) or test_window_hours <= 0:
        raise ValueError("horizon_hours must be a positive integer")
    if mlr is not None:
        if not isinstance(mlr, pd.Series):
            raise TypeError("mlr must be a pandas Series")
        if mlr.index.has_duplicates or not mlr.index.equals(demand_df.index):
            raise ValueError("mlr index must exactly match demand_df.index")
    if mlr_groups is not None:
        if not isinstance(mlr_groups, pd.Series):
            raise TypeError("mlr_groups must be a pandas Series")
        if mlr_groups.index.has_duplicates or not mlr_groups.index.equals(demand_df.index):
            raise ValueError("mlr_groups index must exactly match demand_df.index")
        if mlr_groups.isna().any():
            raise ValueError("mlr_groups must assign every demand timestamp to a bin")

    features = build_features(demand_df, weather_df, horizon=horizon)
    folds = list(
        rolling_origin_splits(
            demand_df.index,
            initial_train_hours=initial_train_hours,
            horizon_hours=test_window_hours,
            step_hours=step_hours,
            include_partial_test=include_partial_test,
            feature_horizon=horizon,
        )
    )
    if not folds:
        raise ValueError("not enough data for one complete backtest fold")

    demand = demand_df["demand_mw"].astype(float)
    first_train_index, _ = folds[0]
    first_fold_training = demand.reindex(first_train_index).to_numpy(dtype=float)
    if len(first_fold_training) <= season:
        raise ValueError(
            "the first fold training series must contain more observations "
            "than the MASE season"
        )
    if not np.isfinite(first_fold_training).all():
        raise ValueError("demand training values must be finite")

    prediction_parts: list[pd.DataFrame] = []
    for train_index, test_index in folds:
        if train_index[-1] >= test_index[0]:
            raise ValueError("training rows must strictly precede the test window")

        train_features = features.reindex(train_index)
        train_targets = demand.reindex(train_index)
        finite_train = train_features.notna().all(axis=1) & train_targets.notna()
        X_train = train_features.loc[finite_train]
        y_train = train_targets.loc[finite_train]
        if X_train.empty:
            raise ValueError(f"no finite-feature training rows before {test_index[0]}")

        X_test = features.reindex(test_index)
        if X_test.isna().any().any():
            raise ValueError(
                f"test features contain missing values in fold starting "
                f"{test_index[0]}"
            )

        seasonal_index = test_index - pd.Timedelta(hours=season)
        seasonal_prediction = demand.reindex(seasonal_index)
        if seasonal_prediction.isna().any():
            raise ValueError(
                f"seasonal-naive history is unavailable for fold starting "
                f"{test_index[0]}"
            )

        models = fit_quantile_models(X_train, y_train)
        intervals = predict_intervals(models, X_test)
        fold_predictions = intervals.copy()
        fold_predictions["actual"] = demand.reindex(test_index).to_numpy(dtype=float)
        fold_predictions["seasonal_naive"] = seasonal_prediction.to_numpy(dtype=float)
        if mlr_groups is not None:
            fold_predictions[SEGMENT_COLUMN] = mlr_groups.reindex(test_index).to_numpy()
        elif mlr is not None:
            mlr_values = mlr.reindex(test_index).to_numpy(dtype=float)
            segment_labels = np.empty(len(test_index), dtype=object)
            segment_labels[:] = None
            segment_labels[mlr_values > 0] = "mlr>0"
            segment_labels[mlr_values == 0] = "mlr==0"
            fold_predictions[SEGMENT_COLUMN] = segment_labels
        prediction_parts.append(fold_predictions)

    pooled = pd.concat(prediction_parts, axis=0)
    if pooled.empty:
        raise ValueError("backtest produced no predictions")

    rows: list[dict[str, object]] = []
    model_names = ("seasonal_naive", "lgbm_quantile")
    if mlr is None and mlr_groups is None:
        for model_name in model_names:
            rows.append(
                _metric_row(
                    model_name,
                    pooled,
                    first_fold_training,
                    season,
                    len(folds),
                )
            )
        return pd.DataFrame(rows, columns=RESULT_COLUMNS)

    segment_labels = pooled[SEGMENT_COLUMN].dropna().drop_duplicates().tolist()
    segments: list[tuple[str, pd.DataFrame]] = [("all", pooled)]
    segments.extend(
        (str(label), pooled.loc[pooled[SEGMENT_COLUMN] == label])
        for label in segment_labels
    )
    for segment, segment_frame in segments:
        for model_name in model_names:
            rows.append(
                {
                    SEGMENT_COLUMN: segment,
                    "n_distinct_days": (
                        pd.DatetimeIndex(segment_frame.index).normalize().nunique()
                    ),
                    **_metric_row(
                        model_name,
                        segment_frame,
                        first_fold_training,
                        season,
                        len(folds),
                    ),
                }
            )
    return pd.DataFrame(
        rows,
        columns=[SEGMENT_COLUMN, "n_distinct_days", *RESULT_COLUMNS],
    )


if __name__ == "__main__":
    synthetic_demand = make_synthetic_demand("2024-01-01", "2024-03-30")
    from src.features.weather import fetch_weather

    synthetic_weather = fetch_weather(
        synthetic_demand.index[0].date().isoformat(),
        synthetic_demand.index[-1].date().isoformat(),
    )
    print("SYNTHETIC DATA, RESULTS MEANINGLESS")
    print(
        run_backtest(
            synthetic_demand,
            synthetic_weather,
            initial_train_hours=1500,
            step_hours=168,
        ).to_string(index=False)
    )
