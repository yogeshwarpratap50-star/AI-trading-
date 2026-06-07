from __future__ import annotations

import pandas as pd

from indicators.base import Indicator


class MovingAverageIndicator(Indicator):
    """Adds EMA and SMA columns used by Phase 2."""

    required_columns = ("close",)

    def add(self, frame: pd.DataFrame) -> pd.DataFrame:
        clean = self._validate(frame)
        close = clean["close"].ffill()
        clean["ema_20"] = close.ewm(span=20, adjust=False).mean()
        clean["ema_50"] = close.ewm(span=50, adjust=False).mean()
        clean["sma_20"] = close.rolling(window=20, min_periods=1).mean()
        clean["sma_50"] = close.rolling(window=50, min_periods=1).mean()
        self.logger.info("Moving averages calculated")
        return clean
