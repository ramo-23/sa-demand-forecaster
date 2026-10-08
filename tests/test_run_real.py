import numpy as np
import pandas as pd
import pytest

from src.evaluate.run_real import (
    DEFAULT_TEST_START,
    MLR_BIN_LABELS,
    _mlr_bins,
    _parse_args,
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
    assert _parse_args([]).test_start == DEFAULT_TEST_START == "2023-04-01"
    assert _parse_args(["--test-start", "2024-10-01"]).test_start == "2024-10-01"
