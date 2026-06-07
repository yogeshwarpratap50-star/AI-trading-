from __future__ import annotations

import pandas as pd

from indicators.base import Indicator


class MACDIndicator(Indicator):
    """Moving Average Convergence Divergence indicator."""

    required_columns = ("close",)

    def __init__(self, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> None:
        super().__init__()
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period

    def add(self, frame: pd.DataFrame) -> pd.DataFrame:
        clean = self._validate(frame)
        close = clean["close"].ffill()
        fast_ema = close.ewm(span=self.fast_period, adjust=False).mean()
        slow_ema = close.ewm(span=self.slow_period, adjust=False).mean()
        clean["macd"] = fast_ema - slow_ema
        clean["macd_signal"] = clean["macd"].ewm(span=self.signal_period, adjust=False).mean()
        clean["macd_histogram"] = clean["macd"] - clean["macd_signal"]
        self.logger.info("MACD calculated")
        return clean
