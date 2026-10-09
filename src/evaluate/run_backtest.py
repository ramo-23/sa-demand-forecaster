from __future__ import annotations

import math
import argparse
from pathlib import Path

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
from src.models.conformal import conformalize_intervals
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
CONFORMAL_MODEL_NAMES = ("lgbm_conformal_90d", "lgbm_conformal_20pct")
RATIO_MODEL_NAMES = ("lgbm_quantile_full_ratio", "lgbm_conformal_90d_ratio")
TARGET_CHOICES = ("level", "ratio_168")


def _split_training_for_calibration(
    train_index: pd.DatetimeIndex,
    calibration_hours: int,
    calibration_fraction: float,
    finite_train_index: pd.DatetimeIndex | None = None,
) -> dict[str, tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
    if not isinstance(calibration_hours, int) or calibration_hours <= 0:
        raise ValueError("calibration_hours must be a positive integer")
    if not np.isfinite(calibration_fraction) or not 0 < calibration_fraction < 1:
        raise ValueError("calibration_fraction must be strictly between 0 and 1")
    eligible_index = train_index if finite_train_index is None else finite_train_index
    if (
        eligible_index.has_duplicates
        or not eligible_index.isin(train_index).all()
        or not eligible_index.is_monotonic_increasing
    ):
        raise ValueError("finite_train_index must be ordered and contained in train_index")
    fraction_hours = max(1, math.ceil(len(eligible_index) * calibration_fraction))
    cal90_window = train_index[-calibration_hours:]
    cal90_index = eligible_index[eligible_index.isin(cal90_window)]
    if len(cal90_window) != calibration_hours or len(cal90_index) != calibration_hours:
        raise ValueError(
            "the final calibration_hours of the training window must all have "
            "finite features for lgbm_conformal_90d"
        )
    fit90_index = eligible_index[eligible_index < cal90_window[0]]
    if fit90_index.empty:
        raise ValueError("training history must precede the 90d calibration block")

    counts = {
        "20pct": fraction_hours,
    }
    splits: dict[str, tuple[pd.DatetimeIndex, pd.DatetimeIndex]] = {}
    splits["90d"] = (fit90_index, cal90_index)
    for name, count in counts.items():
        fit_count = len(eligible_index) - count
        if fit_count <= 0:
            raise ValueError(
                f"training history must exceed the {name} calibration block"
            )
        fit_index = eligible_index[:fit_count]
        calibration_index = eligible_index[-count:]
        if fit_index[-1] >= calibration_index[0]:
            raise ValueError(
                f"{name} calibration rows must strictly follow fit rows"
            )
        splits[name] = (fit_index, calibration_index)
    return splits


def _fold_boundary(
    fold_index: int,
    row_set: str,
    fit_index: pd.DatetimeIndex,
    calibration_index: pd.DatetimeIndex | None,
    test_index: pd.DatetimeIndex,
) -> dict[str, object]:
    record: dict[str, object] = {
        "fold": fold_index,
        "row_set": row_set,
        "fit_end": fit_index[-1],
        "cal_start": None if calibration_index is None else calibration_index[0],
        "cal_end": None if calibration_index is None else calibration_index[-1],
        "test_start": test_index[0],
    }
    if calibration_index is not None:
        fit_end = fit_index[-1]
        cal_start = calibration_index[0]
        cal_end = calibration_index[-1]
        test_start = test_index[0]
        if not fit_end < cal_start <= cal_end < test_start:
            raise ValueError(
                f"fold {fold_index} {row_set} boundaries invalid: "
                f"fit_end={fit_end}, cal_start={cal_start}, "
                f"cal_end={cal_end}, test_start={test_start}"
            )
    return record


def print_fold_boundaries(fold_boundaries: list[dict[str, object]]) -> None:
    if not fold_boundaries:
        raise ValueError("fold_boundaries must contain at least one fold")
    print("First fold boundaries:")
    print(fold_boundaries[0])
    print("Last fold boundaries:")
    print(fold_boundaries[-1])


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
        base_name = model_name.removesuffix("_ratio")
        is_conformal = base_name in CONFORMAL_MODEL_NAMES
        if base_name == "lgbm_quantile_full":
            suffix = "full"
        else:
            suffix = base_name.removeprefix("lgbm_conformal_")
        if model_name.endswith("_ratio"):
            suffix += "_ratio"
        median_column = f"median__{suffix}"
        lower_80_column = f"lower_80__{suffix}"
        upper_80_column = f"upper_80__{suffix}"
        lower_95_column = f"lower_95__{suffix}"
        upper_95_column = f"upper_95__{suffix}"
        predicted = frame[median_column].to_numpy(dtype=float)
        lower_80 = frame[lower_80_column].to_numpy(dtype=float)
        upper_80 = frame[upper_80_column].to_numpy(dtype=float)
        lower_95 = frame[lower_95_column].to_numpy(dtype=float)
        upper_95 = frame[upper_95_column].to_numpy(dtype=float)
        if is_conformal:
            coverage_80 = float(np.mean((actual >= lower_80) & (actual <= upper_80)))
            coverage_95 = float(np.mean((actual >= lower_95) & (actual <= upper_95)))
            width_80 = float(np.mean(np.maximum(upper_80 - lower_80, 0.0)))
            width_95 = float(np.mean(np.maximum(upper_95 - lower_95, 0.0)))
        else:
            coverage_80 = interval_coverage(actual, lower_80, upper_80)
            coverage_95 = interval_coverage(actual, lower_95, upper_95)
            width_80 = mean_interval_width(lower_80, upper_80)
            width_95 = mean_interval_width(lower_95, upper_95)

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


def _scale_intervals(intervals: pd.DataFrame, scale: np.ndarray) -> pd.DataFrame:
    """Scale every predicted quantile by a row-aligned positive denominator."""
    factors = np.asarray(scale, dtype=float)
    if factors.ndim != 1 or len(factors) != len(intervals):
        raise ValueError("scale must be one-dimensional and match interval rows")
    if not np.isfinite(factors).all() or np.any(factors <= 0):
        raise ValueError("scale values must be finite and strictly positive")
    return intervals.mul(factors, axis=0)


def _prediction_artifact(pooled: pd.DataFrame, target: str) -> pd.DataFrame:
    predictions = pd.DataFrame(
        {
            "timestamp": pooled.index,
            "fold": pooled["fold"].to_numpy(dtype=int),
            "y": pooled["actual"].to_numpy(dtype=float),
            "seasonal_naive_point": pooled["seasonal_naive"].to_numpy(dtype=float),
        }
    )
    variants = ["full", "90d", "20pct"]
    if target == "ratio_168":
        variants.extend(("full_ratio", "90d_ratio"))
    for variant in variants:
        model_name = {
            "full": "lgbm_quantile_full",
            "90d": "lgbm_conformal_90d",
            "20pct": "lgbm_conformal_20pct",
            "full_ratio": "lgbm_quantile_full_ratio",
            "90d_ratio": "lgbm_conformal_90d_ratio",
        }[variant]
        predictions[f"{model_name}_point"] = pooled[
            f"median__{variant}"
        ].to_numpy(dtype=float)
        for interval_bound in ("lower_80", "upper_80", "lower_95", "upper_95"):
            predictions[f"{model_name}_{interval_bound}"] = pooled[
                f"{interval_bound}__{variant}"
            ].to_numpy(dtype=float)
    for interval_bound in ("lower_80", "upper_80", "lower_95", "upper_95"):
        predictions[f"seasonal_naive_{interval_bound}"] = np.nan
    return predictions


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
    groupings: dict[str, pd.Series] | None = None,
    calibration_hours: int = 2160,
    calibration_fraction: float = 0.2,
    target: str = "level",
) -> pd.DataFrame:
    """Run pooled rolling-origin evaluation for seasonal-naive and LightGBM.

    ``horizon`` is the feature/model forecast horizon. ``horizon_hours`` sets
    the test-window length and defaults to ``horizon``. The MASE denominator
    is computed from the raw in-sample demand series of the first fold only.
    Each quantile model is fitted only on finite-feature rows before that
    fold's test-window start.

    ``lgbm_quantile_full`` fits on every finite-feature training row.
    ``lgbm_conformal_90d`` fits before the final ``calibration_hours`` finite
    rows and calibrates on those final rows. ``lgbm_conformal_20pct`` fits
    before and calibrates on the final ``calibration_fraction`` of finite
    training rows. Every variant uses the same test rows.
    ``target="ratio_168"`` additionally trains full-fit and 90-day
    conformal models on demand divided by ``demand_lag_168``. Their quantiles
    are converted back to the demand level before conformal calibration and
    scoring; the level models remain present for paired comparison.

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
    if target not in TARGET_CHOICES:
        raise ValueError(f"target must be one of {TARGET_CHOICES}")
    if target == "ratio_168" and horizon > 168:
        raise ValueError("horizon must be <= 168 for target='ratio_168'")
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
    if groupings is not None:
        if not isinstance(groupings, dict) or not groupings:
            raise ValueError("groupings must be a non-empty dict of named Series")
        for name, values in groupings.items():
            if not isinstance(name, str) or not name:
                raise ValueError("grouping names must be non-empty strings")
            if not isinstance(values, pd.Series):
                raise TypeError(f"grouping {name!r} must be a pandas Series")
            if values.index.has_duplicates or not values.index.equals(demand_df.index):
                raise ValueError(
                    f"grouping {name!r} index must exactly match demand_df.index"
                )
            if values.isna().any():
                raise ValueError(f"grouping {name!r} must label every demand timestamp")
    grouping_columns = (
        {
            name: f"__grouping_{position}"
            for position, name in enumerate(groupings or {})
        }
    )

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
    fold_boundaries: list[dict[str, object]] = []
    for fold_number, (train_index, test_index) in enumerate(folds):
        if train_index[-1] >= test_index[0]:
            raise ValueError("training rows must strictly precede the test window")

        train_features = features.reindex(train_index)
        train_targets = demand.reindex(train_index)
        finite_train = train_features.notna().all(axis=1) & train_targets.notna()
        available_train_index = train_features.index[finite_train]
        if available_train_index.empty:
            raise ValueError(f"no finite-feature training rows before {test_index[0]}")
        calibration_splits = _split_training_for_calibration(
            train_index,
            calibration_hours,
            calibration_fraction,
            finite_train_index=available_train_index,
        )
        row_set_boundaries: dict[str, dict[str, object]] = {}
        row_set_boundaries["lgbm_quantile_full"] = _fold_boundary(
            fold_number,
            "lgbm_quantile_full",
            available_train_index,
            None,
            test_index,
        )
        for suffix, (fit_index, calibration_index) in calibration_splits.items():
            row_set = f"lgbm_conformal_{suffix}"
            row_set_boundaries[row_set] = _fold_boundary(
                fold_number,
                row_set,
                fit_index,
                calibration_index,
                test_index,
            )
            if (
                suffix == "90d"
                and test_index[0] - calibration_index[-1]
                != pd.Timedelta(hours=1)
            ):
                raise ValueError(
                    f"fold {fold_number} {row_set} boundaries invalid: "
                    f"cal_end={calibration_index[-1]} must be exactly one hour "
                    f"before test_start={test_index[0]}"
                )
        if target == "ratio_168":
            for base_name, ratio_name in (
                ("lgbm_quantile_full", "lgbm_quantile_full_ratio"),
                ("lgbm_conformal_90d", "lgbm_conformal_90d_ratio"),
            ):
                row_set_boundaries[ratio_name] = {
                    **row_set_boundaries[base_name],
                    "row_set": ratio_name,
                }
        fold_boundaries.append(
            {"fold": fold_number, "row_sets": row_set_boundaries}
        )

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

        fold_predictions = pd.DataFrame(index=test_index)
        fold_predictions["fold"] = fold_number
        fold_predictions["actual"] = demand.reindex(test_index).to_numpy(dtype=float)
        fold_predictions["seasonal_naive"] = seasonal_prediction.to_numpy(dtype=float)

        variant_splits: dict[str, tuple[pd.DatetimeIndex, pd.DatetimeIndex | None]] = {
            "full": (available_train_index, None),
            "90d": calibration_splits["90d"],
            "20pct": calibration_splits["20pct"],
        }
        for variant, (fit_index, calibration_index) in variant_splits.items():
            models = fit_quantile_models(
                train_features.reindex(fit_index),
                train_targets.reindex(fit_index),
            )
            intervals = predict_intervals(models, X_test)
            fold_predictions[f"median__{variant}"] = intervals["median"].to_numpy(
                dtype=float
            )
            for bound in ("lower_80", "upper_80", "lower_95", "upper_95"):
                fold_predictions[f"{bound}__{variant}"] = intervals[bound].to_numpy(
                    dtype=float
                )
            if calibration_index is None:
                continue

            calibration_features = train_features.reindex(calibration_index)
            calibration_intervals = predict_intervals(models, calibration_features)
            calibration_y = demand.reindex(calibration_index).to_numpy(dtype=float)
            lower_80, upper_80 = conformalize_intervals(
                calibration_y,
                calibration_intervals["lower_80"].to_numpy(dtype=float),
                calibration_intervals["upper_80"].to_numpy(dtype=float),
                intervals["lower_80"].to_numpy(dtype=float),
                intervals["upper_80"].to_numpy(dtype=float),
                alpha=0.2,
            )
            lower_95, upper_95 = conformalize_intervals(
                calibration_y,
                calibration_intervals["lower_95"].to_numpy(dtype=float),
                calibration_intervals["upper_95"].to_numpy(dtype=float),
                intervals["lower_95"].to_numpy(dtype=float),
                intervals["upper_95"].to_numpy(dtype=float),
                alpha=0.05,
            )
            fold_predictions[f"lower_80__{variant}"] = lower_80
            fold_predictions[f"upper_80__{variant}"] = upper_80
            fold_predictions[f"lower_95__{variant}"] = lower_95
            fold_predictions[f"upper_95__{variant}"] = upper_95

        if target == "ratio_168":
            ratio_feature = "demand_lag_168"
            train_scale = train_features[ratio_feature].reindex(available_train_index)
            test_scale = X_test[ratio_feature]
            if (
                train_scale.isna().any()
                or test_scale.isna().any()
                or (train_scale <= 0).any()
                or (test_scale <= 0).any()
            ):
                raise ValueError(
                    f"{ratio_feature} must be finite and positive for ratio_168"
                )
            test_scale_values = test_scale.to_numpy(dtype=float)
            for variant, (fit_index, calibration_index) in {
                "full_ratio": (available_train_index, None),
                "90d_ratio": calibration_splits["90d"],
            }.items():
                ratio_fit_scale = train_features[ratio_feature].reindex(fit_index)
                ratio_targets = (
                    train_targets.reindex(fit_index)
                    / ratio_fit_scale.to_numpy(dtype=float)
                )
                ratio_models = fit_quantile_models(
                    train_features.reindex(fit_index),
                    ratio_targets,
                )
                ratio_intervals = _scale_intervals(
                    predict_intervals(ratio_models, X_test),
                    test_scale_values,
                )
                fold_predictions[f"median__{variant}"] = ratio_intervals[
                    "median"
                ].to_numpy(dtype=float)
                for bound in ("lower_80", "upper_80", "lower_95", "upper_95"):
                    fold_predictions[f"{bound}__{variant}"] = ratio_intervals[
                        bound
                    ].to_numpy(dtype=float)
                if calibration_index is None:
                    continue
                cal_scale = train_features[ratio_feature].reindex(calibration_index)
                if cal_scale.isna().any() or (cal_scale <= 0).any():
                    raise ValueError(
                        f"{ratio_feature} must be finite and positive for "
                        "ratio_168 calibration"
                    )
                ratio_calibration_intervals = _scale_intervals(
                    predict_intervals(
                        ratio_models,
                        train_features.reindex(calibration_index),
                    ),
                    cal_scale.to_numpy(dtype=float),
                )
                calibration_y = demand.reindex(calibration_index).to_numpy(dtype=float)
                lower_80, upper_80 = conformalize_intervals(
                    calibration_y,
                    ratio_calibration_intervals["lower_80"].to_numpy(dtype=float),
                    ratio_calibration_intervals["upper_80"].to_numpy(dtype=float),
                    ratio_intervals["lower_80"].to_numpy(dtype=float),
                    ratio_intervals["upper_80"].to_numpy(dtype=float),
                    alpha=0.2,
                )
                lower_95, upper_95 = conformalize_intervals(
                    calibration_y,
                    ratio_calibration_intervals["lower_95"].to_numpy(dtype=float),
                    ratio_calibration_intervals["upper_95"].to_numpy(dtype=float),
                    ratio_intervals["lower_95"].to_numpy(dtype=float),
                    ratio_intervals["upper_95"].to_numpy(dtype=float),
                    alpha=0.05,
                )
                fold_predictions[f"lower_80__{variant}"] = lower_80
                fold_predictions[f"upper_80__{variant}"] = upper_80
                fold_predictions[f"lower_95__{variant}"] = lower_95
                fold_predictions[f"upper_95__{variant}"] = upper_95
        if mlr_groups is not None:
            fold_predictions[SEGMENT_COLUMN] = mlr_groups.reindex(test_index).to_numpy()
        elif mlr is not None:
            mlr_values = mlr.reindex(test_index).to_numpy(dtype=float)
            segment_labels = np.empty(len(test_index), dtype=object)
            segment_labels[:] = None
            segment_labels[mlr_values > 0] = "mlr>0"
            segment_labels[mlr_values == 0] = "mlr==0"
            fold_predictions[SEGMENT_COLUMN] = segment_labels
        for name, column in grouping_columns.items():
            assert groupings is not None
            fold_predictions[column] = groupings[name].reindex(test_index).to_numpy()
        prediction_parts.append(fold_predictions)

    pooled = pd.concat(prediction_parts, axis=0)
    if pooled.empty:
        raise ValueError("backtest produced no predictions")

    rows: list[dict[str, object]] = []
    model_names = ("seasonal_naive", "lgbm_quantile_full", *CONFORMAL_MODEL_NAMES)
    if target == "ratio_168":
        model_names = (*model_names, *RATIO_MODEL_NAMES)
    if mlr is None and mlr_groups is None and groupings is None:
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
        results = pd.DataFrame(rows, columns=RESULT_COLUMNS)
        results.attrs["fold_boundaries"] = fold_boundaries
        results.attrs["predictions"] = _prediction_artifact(pooled, target)
        return results

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
    if groupings is not None:
        for name, column in grouping_columns.items():
            for label in pooled[column].drop_duplicates().tolist():
                segment_frame = pooled.loc[pooled[column] == label]
                for model_name in model_names:
                    rows.append(
                        {
                            "grouping": name,
                            SEGMENT_COLUMN: str(label),
                            "n_distinct_days": (
                                pd.DatetimeIndex(segment_frame.index)
                                .normalize()
                                .nunique()
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
        results = pd.DataFrame(
            rows,
            columns=["grouping", SEGMENT_COLUMN, "n_distinct_days", *RESULT_COLUMNS],
        )
        results.attrs["fold_boundaries"] = fold_boundaries
        results.attrs["predictions"] = _prediction_artifact(pooled, target)
        return results
    results = pd.DataFrame(
        rows,
        columns=[SEGMENT_COLUMN, "n_distinct_days", *RESULT_COLUMNS],
    )
    results.attrs["fold_boundaries"] = fold_boundaries
    results.attrs["predictions"] = _prediction_artifact(pooled, target)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run synthetic demand backtest.")
    parser.add_argument("--target", choices=TARGET_CHOICES, default="level")
    args = parser.parse_args()
    synthetic_demand = make_synthetic_demand("2024-01-01", "2024-03-30")
    from src.features.weather import fetch_weather

    synthetic_weather = fetch_weather(
        synthetic_demand.index[0].date().isoformat(),
        synthetic_demand.index[-1].date().isoformat(),
    )
    print("SYNTHETIC DATA, RESULTS MEANINGLESS")
    results = run_backtest(
        synthetic_demand,
        synthetic_weather,
        initial_train_hours=1500,
        step_hours=168,
        calibration_hours=216,
        calibration_fraction=0.2,
        target=args.target,
    )
    print_fold_boundaries(results.attrs["fold_boundaries"])
    print(results.to_string(index=False))
    Path("results").mkdir(parents=True, exist_ok=True)
    results.attrs["predictions"].to_csv(
        Path("results") / f"backtest_predictions_{args.target}.csv",
        index=False,
    )
