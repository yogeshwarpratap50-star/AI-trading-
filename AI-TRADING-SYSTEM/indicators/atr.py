from __future__ import annotations

import pandas as pd

from indicators.base import Indicator


class ATRIndicator(Indicator):
    """Average True Range."""

    required_columns = ("high", "low", "close")

    def __init__(self, period: int = 14) -> None:
        super().__init__()
        self.period = period

    def add(self, frame: pd.DataFrame) -> pd.DataFrame:
        clean = self._validate(frame)
        high = clean["high"].ffill()
        low = clean["low"].ffill()
        close = clean["close"].ffill()
        previous_close = close.shift(1)
        true_range = pd.concat(
            [
                high - low,
                (high - previous_close).abs(),
                (low - previous_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        clean["atr_14"] = true_range.ewm(alpha=1 / self.period, min_periods=1, adjust=False).mean()
        self.logger.info("ATR calculated", extra={"period": self.period})
        return clean
