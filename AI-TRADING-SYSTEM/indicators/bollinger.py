from __future__ import annotations

import pandas as pd

from indicators.base import Indicator


class BollingerBandsIndicator(Indicator):
    """Bollinger Bands and band width."""

    required_columns = ("close",)

    def __init__(self, period: int = 20, standard_deviations: float = 2.0) -> None:
        super().__init__()
        self.period = period
        self.standard_deviations = standard_deviations

    def add(self, frame: pd.DataFrame) -> pd.DataFrame:
        clean = self._validate(frame)
        close = clean["close"].ffill()
        middle = close.rolling(window=self.period, min_periods=1).mean()
        std = close.rolling(window=self.period, min_periods=1).std(ddof=0)
        clean["bb_middle"] = middle
        clean["bb_upper"] = middle + self.standard_deviations * std
        clean["bb_lower"] = middle - self.standard_deviations * std
        clean["bb_width"] = ((clean["bb_upper"] - clean["bb_lower"]) / middle.replace(0, pd.NA)).fillna(0)
        self.logger.info("Bollinger Bands calculated")
        return clean
