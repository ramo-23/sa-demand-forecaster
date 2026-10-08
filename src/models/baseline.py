from collections.abc import Sequence

import numpy as np


def seasonal_naive_forecast(
    series: Sequence[float] | np.ndarray,
    horizon: int,
    season: int = 168,
) -> np.ndarray:
    """Forecast by repeating the latest complete observed seasonal cycle."""
    values = np.asarray(series, dtype=float)
    if values.ndim != 1:
        raise ValueError("series must be one-dimensional")
    if not np.isfinite(values).all():
        raise ValueError("series must contain only finite values")
    if not isinstance(horizon, int) or horizon <= 0:
        raise ValueError("horizon must be a positive integer")
    if not isinstance(season, int) or season <= 0:
        raise ValueError("season must be a positive integer")
    if len(values) < season:
        raise ValueError("series must contain at least one complete season")

    last_season = values[-season:]
    repeats = (horizon + season - 1) // season
    return np.tile(last_season, repeats)[:horizon].copy()
