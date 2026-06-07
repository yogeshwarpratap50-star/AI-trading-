import pandas as pd

from training.dataset_builder import MODEL_FEATURES, DatasetBuilder


def sample_ohlcv(rows: int = 80) -> pd.DataFrame:
    close = pd.Series([100 + index * 0.4 + (index % 5) * 0.2 for index in range(rows)], dtype="float")
    return pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=rows, freq="D"),
            "open": close - 0.15,
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "volume": [1000 + index * 11 for index in range(rows)],
        }
    )


def test_dataset_builder_creates_model_features_and_labels() -> None:
    sentiment = pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=5, freq="D"),
            "sentiment_score": [0.2, 0.1, -0.1, 0.3, 0.0],
            "confidence_score": [0.8, 0.7, 0.6, 0.9, 0.5],
        }
    )

    dataset = DatasetBuilder().build(sample_ohlcv(), sentiment=sentiment)

    assert set(MODEL_FEATURES).issubset(dataset.columns)
    assert "target_next_day_up" in dataset.columns
    assert not dataset[MODEL_FEATURES].isna().any().any()
