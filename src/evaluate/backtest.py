from collections.abc import Iterator

import pandas as pd


def rolling_origin_splits(
    index: pd.DatetimeIndex,
    initial_train_hours: int,
    horizon_hours: int,
    step_hours: int,
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
    ):
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if len(index) > 1:
        differences = index[1:] - index[:-1]
        if not (differences == pd.Timedelta(hours=1)).all():
            raise ValueError("index must have consecutive hourly timestamps")

    train_end = initial_train_hours
    while train_end + horizon_hours <= len(index):
        test_end = train_end + horizon_hours
        yield index[:train_end], index[train_end:test_end]
        train_end += step_hours
