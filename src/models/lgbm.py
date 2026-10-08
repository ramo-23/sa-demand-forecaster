from collections.abc import Sequence

import lightgbm as lgb
import numpy as np
import pandas as pd

DEFAULT_QUANTILES = (0.025, 0.1, 0.5, 0.9, 0.975)
INTERVAL_COLUMNS = (
    "lower_95",
    "lower_80",
    "median",
    "upper_80",
    "upper_95",
)


def _validate_features(X: pd.DataFrame, name: str) -> None:
    if not isinstance(X, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame")
    if X.empty or X.shape[1] == 0:
        raise ValueError(f"{name} must contain rows and feature columns")
    if not X.columns.is_unique:
        raise ValueError(f"{name} feature columns must be unique")
    if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in X.dtypes):
        raise ValueError(f"{name} features must all be numeric")
    values = X.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"{name} features must contain only finite values")


def fit_quantile_models(
    X_train: pd.DataFrame,
    y_train: Sequence[float] | np.ndarray | pd.Series,
    quantiles: tuple[float, ...] = DEFAULT_QUANTILES,
) -> dict[float, lgb.LGBMRegressor]:
    """Fit one LightGBM quantile regressor per requested quantile.

    Each model is fitted exclusively on the supplied training rows. No
    preprocessing state is learned outside those rows.
    """
    _validate_features(X_train, "X_train")
    targets = np.asarray(y_train, dtype=float)
    if targets.ndim != 1 or len(targets) != len(X_train):
        raise ValueError("y_train must be one-dimensional and match X_train rows")
    if not np.isfinite(targets).all():
        raise ValueError("y_train must contain only finite values")
    if not quantiles:
        raise ValueError("quantiles must not be empty")
    if len(set(quantiles)) != len(quantiles):
        raise ValueError("quantiles must be unique")
    if any(not np.isfinite(q) or q <= 0 or q >= 1 for q in quantiles):
        raise ValueError("each quantile must be strictly between 0 and 1")

    models: dict[float, lgb.LGBMRegressor] = {}
    for quantile in quantiles:
        model = lgb.LGBMRegressor(
            objective="quantile",
            alpha=quantile,
            n_estimators=100,
            learning_rate=0.05,
            num_leaves=15,
            min_child_samples=10,
            random_state=42,
            n_jobs=1,
            verbosity=-1,
        )
        model.fit(X_train, targets)
        models[float(quantile)] = model
    return models


def predict_intervals(
    models: dict[float, lgb.LGBMRegressor],
    X: pd.DataFrame,
) -> pd.DataFrame:
    """Predict and row-sort 80% and 95% quantile interval bounds."""
    _validate_features(X, "X")
    missing_quantiles = [q for q in DEFAULT_QUANTILES if q not in models]
    if missing_quantiles:
        raise ValueError(
            "models must contain quantiles "
            f"{', '.join(str(q) for q in missing_quantiles)}"
        )

    predictions = np.column_stack(
        [np.asarray(models[q].predict(X), dtype=float) for q in DEFAULT_QUANTILES]
    )
    if predictions.shape != (len(X), len(DEFAULT_QUANTILES)):
        raise ValueError("quantile models returned predictions with invalid shapes")
    if not np.isfinite(predictions).all():
        raise ValueError("quantile models returned non-finite predictions")

    ordered = np.sort(predictions, axis=1)
    result = pd.DataFrame(ordered, index=X.index, columns=INTERVAL_COLUMNS)
    result.index.name = X.index.name
    return result
