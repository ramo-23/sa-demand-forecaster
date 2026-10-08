import numpy as np
import pandas as pd

import src.data.synthetic as synthetic
from src.features.build import build_features
from src.models.lgbm import INTERVAL_COLUMNS, fit_quantile_models, predict_intervals

TIMEZONE = "Africa/Johannesburg"


def _make_training_data(
    monkeypatch,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    def fake_fetch_weather(start_date: str, end_date: str) -> pd.DataFrame:
        index = pd.date_range(
            start=start_date,
            end=f"{end_date} 23:00",
            freq="h",
            tz=TIMEZONE,
            name="timestamp",
        )
        hours = np.arange(len(index), dtype=float)
        return pd.DataFrame(
            {
                "temp_weighted": 18.0 + 8 * np.sin(hours / (24 * 30)),
                "temp_johannesburg": 19.0 + 9 * np.sin(hours / (24 * 30)),
                "temp_cape_town": 17.0 + 7 * np.sin(hours / (24 * 30)),
                "temp_durban": 21.0 + 6 * np.sin(hours / (24 * 30)),
                "temp_bloemfontein": 16.0 + 10 * np.sin(hours / (24 * 30)),
            },
            index=index,
        )

    monkeypatch.setattr(synthetic, "fetch_weather", fake_fetch_weather)
    demand = synthetic.make_synthetic_demand("2024-01-01", "2024-02-10", seed=23)
    weather = fake_fetch_weather("2024-01-01", "2024-02-10")
    all_features = build_features(demand, weather)

    target = demand["demand_mw"].reindex(all_features.index)
    usable = all_features.notna().all(axis=1) & target.notna()
    features = all_features.loc[usable]
    targets = target.loc[usable]
    split_at = int(len(features) * 0.8)
    return features.iloc[:split_at], targets.iloc[:split_at], features.iloc[split_at:]


def test_fit_and_predict_quantile_models_on_synthetic_data(monkeypatch) -> None:
    X_train, y_train, X_test = _make_training_data(monkeypatch)

    models = fit_quantile_models(X_train, y_train)
    predictions = predict_intervals(models, X_test.iloc[:48])

    assert predictions.columns.tolist() == list(INTERVAL_COLUMNS)
    assert len(models) == 5
    assert len(predictions) == 48
    assert (
        predictions["lower_95"]
        .le(predictions["lower_80"])
        .all()
    )
    assert predictions["lower_80"].le(predictions["median"]).all()
    assert predictions["median"].le(predictions["upper_80"]).all()
    assert predictions["upper_80"].le(predictions["upper_95"]).all()


def test_predict_preserves_input_and_index(monkeypatch) -> None:
    X_train, y_train, X_test = _make_training_data(monkeypatch)
    models = fit_quantile_models(X_train, y_train)
    X = X_test.iloc[:24].copy()
    original = X.copy(deep=True)

    predictions = predict_intervals(models, X)

    pd.testing.assert_frame_equal(X, original)
    assert predictions.index.equals(X.index)
    assert predictions.index.name == X.index.name
