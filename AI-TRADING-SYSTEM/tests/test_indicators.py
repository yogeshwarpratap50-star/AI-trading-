import pandas as pd

from indicators.atr import ATRIndicator
from indicators.macd import MACDIndicator
from indicators.moving_averages import MovingAverageIndicator
from indicators.rsi import RSIIndicator


def sample_frame(rows: int = 60) -> pd.DataFrame:
    close = pd.Series(range(100, 100 + rows), dtype="float")
    return pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=rows, freq="D"),
            "open": close - 0.5,
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "volume": [1000 + index for index in range(rows)],
        }
    )


def test_rsi_for_uptrend_is_high() -> None:
    result = RSIIndicator().add(sample_frame())

    assert "rsi_14" in result.columns
    assert result["rsi_14"].between(0, 100).all()
    assert result["rsi_14"].iloc[-1] > 90


def test_macd_columns_are_added() -> None:
    result = MACDIndicator().add(sample_frame())

    assert {"macd", "macd_signal", "macd_histogram"}.issubset(result.columns)
    assert result["macd"].iloc[-1] > 0


def test_moving_averages_match_expected_short_window_mean() -> None:
    result = MovingAverageIndicator().add(sample_frame())

    assert {"ema_20", "ema_50", "sma_20", "sma_50"}.issubset(result.columns)
    assert result["sma_20"].iloc[-1] == sum(range(140, 160)) / 20


def test_atr_is_positive() -> None:
    result = ATRIndicator().add(sample_frame())

    assert "atr_14" in result.columns
    assert result["atr_14"].iloc[-1] > 0
