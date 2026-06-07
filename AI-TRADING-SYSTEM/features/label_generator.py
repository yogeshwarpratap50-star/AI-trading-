from __future__ import annotations

import logging

import pandas as pd


class LabelGenerator:
    """Generates next-day and intraday direction labels."""

    def __init__(self, buy_threshold: float = 0.0025, sell_threshold: float = -0.0025) -> None:
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.logger = logging.getLogger(self.__class__.__name__)

    def add_labels(self, frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            raise ValueError("Label input DataFrame cannot be empty.")
        required = {"open", "close"}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"Missing label input columns: {sorted(missing)}")

        labeled = frame.copy()
        close = pd.to_numeric(labeled["close"], errors="coerce").ffill()
        open_price = pd.to_numeric(labeled["open"], errors="coerce").ffill()
        next_close = close.shift(-1)
        labeled["target_next_day_up"] = (next_close > close).astype("Int64")
        labeled.loc[next_close.isna(), "target_next_day_up"] = pd.NA

        intraday_return = ((close - open_price) / open_price.replace(0, pd.NA)).fillna(0)
        labeled["target_intraday_direction"] = "HOLD"
        labeled.loc[intraday_return >= self.buy_threshold, "target_intraday_direction"] = "BUY"
        labeled.loc[intraday_return <= self.sell_threshold, "target_intraday_direction"] = "SELL"
        self.logger.info("Labels generated", extra={"rows": len(labeled)})
        return labeled
