from __future__ import annotations

import pandas as pd

from indicators.base import Indicator


class VWAPIndicator(Indicator):
    """Volume Weighted Average Price plus volume ratio and daily returns."""

    required_columns = ("high", "low", "close", "volume")

    def add(self, frame: pd.DataFrame) -> pd.DataFrame:
        clean = self._validate(frame)
        high = clean["high"].ffill()
        low = clean["low"].ffill()
        close = clean["close"].ffill()
        volume = clean["volume"].fillna(0).clip(lower=0)
        typical_price = (high + low + close) / 3
        cumulative_volume = volume.cumsum().replace(0, pd.NA)
        clean["vwap"] = ((typical_price * volume).cumsum() / cumulative_volume).ffill().fillna(close)
        average_volume = volume.rolling(window=20, min_periods=1).mean().replace(0, pd.NA)
        clean["volume_ratio"] = (volume / average_volume).fillna(0)
        clean["daily_return"] = close.pct_change().fillna(0)
        self.logger.info("VWAP, volume ratio, and daily return calculated")
        return clean
