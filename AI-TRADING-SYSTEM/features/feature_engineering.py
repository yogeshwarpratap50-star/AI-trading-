from __future__ import annotations

import logging

import pandas as pd

from features.data_validation import DataQualityValidator, ValidationReport
from indicators.atr import ATRIndicator
from indicators.bollinger import BollingerBandsIndicator
from indicators.macd import MACDIndicator
from indicators.moving_averages import MovingAverageIndicator
from indicators.rsi import RSIIndicator
from indicators.vwap import VWAPIndicator


FEATURE_COLUMNS = [
    "rsi_14",
    "macd",
    "macd_signal",
    "macd_histogram",
    "ema_20",
    "ema_50",
    "ema_difference",
    "sma_20",
    "sma_50",
    "sma_difference",
    "atr_14",
    "bb_width",
    "volume_ratio",
    "daily_return",
    "price_change_pct",
    "gap_up_down_pct",
    "volatility",
]


class FeatureEngineeringService:
    """Builds model-ready technical feature datasets."""

    def __init__(self, validator: DataQualityValidator | None = None) -> None:
        self.validator = validator or DataQualityValidator()
        self.indicators = [
            RSIIndicator(),
            MACDIndicator(),
            MovingAverageIndicator(),
            ATRIndicator(),
            BollingerBandsIndicator(),
            VWAPIndicator(),
        ]
        self.logger = logging.getLogger(self.__class__.__name__)

    def create_feature_dataset(self, frame: pd.DataFrame) -> tuple[pd.DataFrame, ValidationReport]:
        report = self.validator.validate(frame)
        dataset = self._prepare(frame)
        for indicator in self.indicators:
            dataset = indicator.add(dataset)

        close = dataset["close"].ffill()
        open_price = dataset["open"].ffill()
        previous_close = close.shift(1)
        dataset["ema_difference"] = dataset["ema_20"] - dataset["ema_50"]
        dataset["sma_difference"] = dataset["sma_20"] - dataset["sma_50"]
        dataset["price_change_pct"] = close.pct_change().fillna(0)
        dataset["gap_up_down_pct"] = ((open_price - previous_close) / previous_close.replace(0, pd.NA)).fillna(0)
        dataset["volatility"] = dataset["daily_return"].rolling(window=20, min_periods=1).std(ddof=0).fillna(0)

        ordered_columns = [column for column in ["date", "open", "high", "low", "close", "volume"] if column in dataset.columns]
        feature_dataset = dataset[ordered_columns + FEATURE_COLUMNS].copy()
        feature_dataset[FEATURE_COLUMNS] = feature_dataset[FEATURE_COLUMNS].replace([float("inf"), float("-inf")], pd.NA)
        feature_dataset[FEATURE_COLUMNS] = feature_dataset[FEATURE_COLUMNS].ffill().fillna(0)
        self.logger.info("Feature dataset created", extra={"rows": len(feature_dataset)})
        return feature_dataset, report

    @staticmethod
    def _prepare(frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            raise ValueError("Feature input DataFrame cannot be empty.")
        dataset = frame.copy()
        dataset.columns = [str(column).lower().replace(" ", "_") for column in dataset.columns]
        if "date" in dataset.columns:
            dataset["date"] = pd.to_datetime(dataset["date"], errors="coerce")
            dataset = dataset.sort_values("date").drop_duplicates(subset=["date"])
        for column in ("open", "high", "low", "close", "volume"):
            if column not in dataset.columns:
                raise ValueError(f"Missing required feature input column: {column}")
            dataset[column] = pd.to_numeric(dataset[column], errors="coerce")
        dataset[["open", "high", "low", "close"]] = dataset[["open", "high", "low", "close"]].ffill().bfill()
        dataset["volume"] = dataset["volume"].fillna(0).clip(lower=0)
        return dataset.reset_index(drop=True)
