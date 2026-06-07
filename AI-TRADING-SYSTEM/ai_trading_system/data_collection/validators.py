from __future__ import annotations

import pandas as pd

from ai_trading_system.data_collection.exceptions import DataValidationError


REQUIRED_OHLCV_COLUMNS = ("date", "open", "high", "low", "close", "volume")


class OHLCVValidator:
    """Validates normalized OHLCV data frames."""

    def validate(self, frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            raise DataValidationError("OHLCV data frame is empty.")

        missing = set(REQUIRED_OHLCV_COLUMNS).difference(frame.columns)
        if missing:
            raise DataValidationError(f"Missing required OHLCV columns: {sorted(missing)}")

        clean = frame.copy()
        clean["date"] = pd.to_datetime(clean["date"], utc=True, errors="coerce")
        numeric_columns = ["open", "high", "low", "close", "volume"]
        for column in numeric_columns:
            clean[column] = pd.to_numeric(clean[column], errors="coerce")

        clean = clean.dropna(subset=["date", "open", "high", "low", "close"])
        clean["volume"] = clean["volume"].fillna(0)
        clean = clean[clean["high"] >= clean["low"]]
        clean = clean[clean["volume"] >= 0]
        clean = clean.sort_values("date").drop_duplicates(subset=["date"])

        if clean.empty:
            raise DataValidationError("OHLCV data is invalid after cleaning.")

        return clean.reset_index(drop=True)
