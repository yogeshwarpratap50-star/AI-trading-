import pandas as pd

from features.feature_engineering import FEATURE_COLUMNS, FeatureEngineeringService
from features.label_generator import LabelGenerator


def sample_ohlcv() -> pd.DataFrame:
    close = pd.Series([100 + index * 0.5 for index in range(80)], dtype="float")
    return pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=80, freq="D"),
            "open": close - 0.25,
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "volume": [1000 + index * 10 for index in range(80)],
        }
    )


def test_feature_pipeline_generates_expected_features() -> None:
    features, report = FeatureEngineeringService().create_feature_dataset(sample_ohlcv())

    assert report.row_count == 80
    assert set(FEATURE_COLUMNS).issubset(features.columns)
    assert not features[FEATURE_COLUMNS].isna().any().any()
    assert features["ema_difference"].iloc[-1] > 0


def test_label_generator_creates_next_day_and_intraday_targets() -> None:
    labeled = LabelGenerator().add_labels(sample_ohlcv())

    assert "target_next_day_up" in labeled.columns
    assert "target_intraday_direction" in labeled.columns
    assert labeled["target_next_day_up"].dropna().eq(1).all()
    assert set(labeled["target_intraday_direction"].unique()).issubset({"BUY", "SELL", "HOLD"})
