from collections.abc import Iterator

import pandas as pd


def rolling_origin_splits(
    index: pd.DatetimeIndex,
    initial_train_hours: int,
    horizon_hours: int,
    step_hours: int,
    include_partial_test: bool = False,
    feature_horizon: int = 24,
) -> Iterator[tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
    """Yield expanding chronological train/test index labels for hourly data.

    Test windows are advanced by step_hours and may overlap one another when
    step_hours is smaller than horizon_hours. Train and test are disjoint in
    every split.
    """
    if not isinstance(index, pd.DatetimeIndex):
        raise TypeError("index must be a pandas DatetimeIndex")
    if index.has_duplicates or not index.is_monotonic_increasing:
        raise ValueError("index must be strictly increasing with no duplicates")
    for name, value in (
        ("initial_train_hours", initial_train_hours),
        ("horizon_hours", horizon_hours),
        ("step_hours", step_hours),
        ("feature_horizon", feature_horizon),
    ):
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if len(index) > 1:
        differences = index[1:] - index[:-1]
        if not (differences == pd.Timedelta(hours=1)).all():
            raise ValueError("index must have consecutive hourly timestamps")

    minimum_train_hours = 336 + (feature_horizon - 24) + 168
    if initial_train_hours <= minimum_train_hours:
        raise ValueError(
            "initial_train_hours must exceed "
            f"{minimum_train_hours} to provide finite lag and rolling features"
        )

    if not isinstance(include_partial_test, bool):
        raise ValueError("include_partial_test must be a boolean")

    train_end = initial_train_hours
    while train_end < len(index):
        test_end = min(train_end + horizon_hours, len(index))
        if test_end - train_end < horizon_hours and not include_partial_test:
            break
        yield index[:train_end], index[train_end:test_end]
        if test_end == len(index):
            break
        train_end += step_hours
