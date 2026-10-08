import pandas as pd
import pytest

from src.evaluate.backtest import rolling_origin_splits


def test_rolling_origin_splits_are_expanding_ordered_and_disjoint() -> None:
    index = pd.date_range(
        "2024-01-01",
        periods=600,
        freq="h",
        tz="Africa/Johannesburg",
    )

    splits = list(
        rolling_origin_splits(
            index,
            initial_train_hours=505,
            horizon_hours=24,
            step_hours=24,
        )
    )

    assert len(splits) == 3
    previous_train_length = 0
    previous_test_start = None
    for train_idx, test_idx in splits:
        assert len(train_idx) >= previous_train_length
        if previous_train_length:
            assert len(train_idx) > previous_train_length
        assert train_idx[-1] < test_idx[0]
        assert not train_idx.intersection(test_idx).size
        assert (test_idx > train_idx[-1]).all()
        if previous_test_start is not None:
            assert test_idx[0] > previous_test_start
        previous_train_length = len(train_idx)
        previous_test_start = test_idx[0]


def test_rolling_origin_requires_regular_hourly_index() -> None:
    index = pd.DatetimeIndex(
        ["2024-01-01 00:00", "2024-01-01 02:00"],
        tz="Africa/Johannesburg",
    )

    with pytest.raises(ValueError, match="consecutive hourly"):
        list(rolling_origin_splits(index, 1, 1, 1))


def test_rolling_origin_rejects_insufficient_initial_training_hours() -> None:
    index = pd.date_range(
        "2024-01-01",
        periods=600,
        freq="h",
        tz="Africa/Johannesburg",
    )

    with pytest.raises(ValueError, match="must exceed 504"):
        list(
            rolling_origin_splits(
                index,
                initial_train_hours=504,
                horizon_hours=24,
                step_hours=24,
            )
        )


def test_rolling_origin_can_include_final_partial_test_window() -> None:
    index = pd.date_range(
        "2024-01-01",
        periods=530,
        freq="h",
        tz="Africa/Johannesburg",
    )

    splits = list(
        rolling_origin_splits(
            index,
            initial_train_hours=505,
            horizon_hours=24,
            step_hours=24,
            include_partial_test=True,
        )
    )

    assert [len(test_idx) for _, test_idx in splits] == [24, 1]
    assert splits[-1][1][-1] == index[-1]
