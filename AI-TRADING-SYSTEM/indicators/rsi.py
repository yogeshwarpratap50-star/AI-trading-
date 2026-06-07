from __future__ import annotations

import pandas as pd

from indicators.base import Indicator


class RSIIndicator(Indicator):
    """Relative Strength Index using Wilder's smoothing."""

    required_columns = ("close",)

    def __init__(self, period: int = 14) -> None:
        super().__init__()
        self.period = period

    def add(self, frame: pd.DataFrame) -> pd.DataFrame:
        clean = self._validate(frame)
        close = clean["close"].ffill()
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        average_gain = gain.ewm(alpha=1 / self.period, min_periods=self.period, adjust=False).mean()
        average_loss = loss.ewm(alpha=1 / self.period, min_periods=self.period, adjust=False).mean()
        relative_strength = average_gain / average_loss.replace(0, pd.NA)
        rsi = 100 - (100 / (1 + relative_strength))
        rsi = rsi.mask((average_loss == 0) & (average_gain > 0), 100.0)
        rsi = rsi.mask((average_gain == 0) & (average_loss > 0), 0.0)
        clean["rsi_14"] = rsi.fillna(50.0).clip(0, 100)
        self.logger.info("RSI calculated", extra={"period": self.period})
        return clean
