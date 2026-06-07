from __future__ import annotations

import logging

import pandas as pd

from features.feature_engineering import FEATURE_COLUMNS, FeatureEngineeringService
from features.label_generator import LabelGenerator


TECHNICAL_MODEL_FEATURES = [
    "rsi_14",
    "macd",
    "macd_signal",
    "macd_histogram",
    "ema_20",
    "ema_50",
    "sma_20",
    "sma_50",
    "atr_14",
    "bb_width",
    "vwap",
    "volume_ratio",
    "daily_return",
]

SENTIMENT_MODEL_FEATURES = ["sentiment_score", "news_count", "confidence_score"]
MARKET_MODEL_FEATURES = ["nifty_trend_score", "sector_trend_score", "market_health_score"]
MODEL_FEATURES = TECHNICAL_MODEL_FEATURES + SENTIMENT_MODEL_FEATURES + MARKET_MODEL_FEATURES


class DatasetBuilder:
    """Builds model-ready datasets from OHLCV, sentiment, and market context."""

    def __init__(self) -> None:
        self.feature_service = FeatureEngineeringService()
        self.label_generator = LabelGenerator()
        self.logger = logging.getLogger(self.__class__.__name__)

    def build(
        self,
        ohlcv: pd.DataFrame,
        sentiment: pd.DataFrame | None = None,
        market_context: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        features, report = self.feature_service.create_feature_dataset(ohlcv)
        dataset = self._add_vwap_if_missing(features)
        dataset = self._merge_sentiment(dataset, sentiment)
        dataset = self._merge_market_context(dataset, market_context)
        dataset = self.label_generator.add_labels(dataset)
        dataset = dataset.dropna(subset=["target_next_day_up"]).reset_index(drop=True)
        dataset[MODEL_FEATURES] = dataset[MODEL_FEATURES].replace([float("inf"), float("-inf")], pd.NA)
        dataset[MODEL_FEATURES] = dataset[MODEL_FEATURES].ffill().fillna(0)
        self.logger.info("Training dataset built", extra={"rows": len(dataset), "valid_input": report.is_valid})
        return dataset

    @staticmethod
    def _add_vwap_if_missing(dataset: pd.DataFrame) -> pd.DataFrame:
        if "vwap" in dataset.columns:
            return dataset
        enriched = dataset.copy()
        typical_price = (enriched["high"] + enriched["low"] + enriched["close"]) / 3
        cumulative_volume = enriched["volume"].cumsum().replace(0, pd.NA)
        enriched["vwap"] = ((typical_price * enriched["volume"]).cumsum() / cumulative_volume).ffill().fillna(enriched["close"])
        return enriched

    @staticmethod
    def _merge_sentiment(dataset: pd.DataFrame, sentiment: pd.DataFrame | None) -> pd.DataFrame:
        enriched = dataset.copy()
        if sentiment is None or sentiment.empty:
            enriched["sentiment_score"] = 0.0
            enriched["news_count"] = 0.0
            enriched["confidence_score"] = 0.0
            return enriched

        sentiment_data = sentiment.copy()
        sentiment_data["date"] = pd.to_datetime(
            sentiment_data.get("date", sentiment_data.get("created_at")),
            errors="coerce",
        ).dt.normalize()
        grouped = sentiment_data.groupby("date").agg(
            sentiment_score=("sentiment_score", "mean"),
            news_count=("sentiment_score", "size"),
            confidence_score=("confidence_score", "mean"),
        )
        enriched["date_key"] = pd.to_datetime(enriched["date"], errors="coerce").dt.normalize()
        enriched = enriched.merge(grouped, left_on="date_key", right_index=True, how="left").drop(columns=["date_key"])
        enriched[["sentiment_score", "news_count", "confidence_score"]] = (
            enriched[["sentiment_score", "news_count", "confidence_score"]].fillna(0)
        )
        return enriched

    @staticmethod
    def _merge_market_context(dataset: pd.DataFrame, market_context: pd.DataFrame | None) -> pd.DataFrame:
        enriched = dataset.copy()
        if market_context is None or market_context.empty:
            enriched["nifty_trend_score"] = 0.0
            enriched["sector_trend_score"] = 0.0
            enriched["market_health_score"] = 50.0
            return enriched

        context = market_context.copy()
        context["date"] = pd.to_datetime(context["date"], errors="coerce").dt.normalize()
        enriched["date_key"] = pd.to_datetime(enriched["date"], errors="coerce").dt.normalize()
        enriched = enriched.merge(context, left_on="date_key", right_on="date", how="left", suffixes=("", "_market"))
        enriched = enriched.drop(columns=[column for column in ["date_key", "date_market"] if column in enriched.columns])
        for column, default in {
            "nifty_trend_score": 0.0,
            "sector_trend_score": 0.0,
            "market_health_score": 50.0,
        }.items():
            if column not in enriched.columns:
                enriched[column] = default
            enriched[column] = pd.to_numeric(enriched[column], errors="coerce").fillna(default)
        return enriched

    @staticmethod
    def feature_columns() -> list[str]:
        return MODEL_FEATURES.copy()
