from __future__ import annotations

from dataclasses import dataclass
import logging

import pandas as pd


@dataclass(frozen=True)
class MarketIntelligenceReport:
    nifty_trend: str
    sector_trend: dict[str, str]
    top_gainers: list[str]
    top_losers: list[str]
    high_volume_stocks: list[str]
    volatility: float
    gap_up_stocks: list[str]
    gap_down_stocks: list[str]
    market_health_score: float


class MarketIntelligenceService:
    """Creates market breadth and health summaries from OHLCV snapshots."""

    def __init__(self) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)

    def generate(self, frame: pd.DataFrame, sector_map: dict[str, str] | None = None, nifty_symbol: str = "^NSEI") -> MarketIntelligenceReport:
        if frame.empty:
            raise ValueError("Market intelligence input cannot be empty.")
        data = frame.copy()
        required = {"symbol", "open", "close", "previous_close", "volume"}
        missing = required.difference(data.columns)
        if missing:
            raise ValueError(f"Missing market intelligence columns: {sorted(missing)}")

        for column in ["open", "close", "previous_close", "volume"]:
            data[column] = pd.to_numeric(data[column], errors="coerce")
        data = data.dropna(subset=["open", "close", "previous_close"])
        data["return_pct"] = (data["close"] - data["previous_close"]) / data["previous_close"].replace(0, pd.NA)
        data["gap_pct"] = (data["open"] - data["previous_close"]) / data["previous_close"].replace(0, pd.NA)
        data["volume_ratio"] = data["volume"] / data["volume"].rolling(window=min(20, len(data)), min_periods=1).mean().replace(0, pd.NA)

        nifty_trend = self._trend_for_symbol(data, nifty_symbol)
        top_gainers = data.sort_values("return_pct", ascending=False)["symbol"].head(5).astype(str).tolist()
        top_losers = data.sort_values("return_pct", ascending=True)["symbol"].head(5).astype(str).tolist()
        high_volume = data.sort_values("volume_ratio", ascending=False)["symbol"].head(5).astype(str).tolist()
        gap_up = data.loc[data["gap_pct"] > 0.005, "symbol"].astype(str).tolist()
        gap_down = data.loc[data["gap_pct"] < -0.005, "symbol"].astype(str).tolist()
        volatility = float(data["return_pct"].std(ddof=0) or 0)
        sector_trend = self._sector_trend(data, sector_map or {})
        health = self._health_score(data, volatility)
        self.logger.info("Market intelligence report generated", extra={"market_health_score": health})
        return MarketIntelligenceReport(
            nifty_trend=nifty_trend,
            sector_trend=sector_trend,
            top_gainers=top_gainers,
            top_losers=top_losers,
            high_volume_stocks=high_volume,
            volatility=round(volatility, 4),
            gap_up_stocks=gap_up,
            gap_down_stocks=gap_down,
            market_health_score=health,
        )

    @staticmethod
    def _trend_for_symbol(data: pd.DataFrame, symbol: str) -> str:
        row = data.loc[data["symbol"] == symbol]
        if row.empty:
            average_return = data["return_pct"].mean()
        else:
            average_return = row["return_pct"].iloc[-1]
        if average_return > 0.003:
            return "Bullish"
        if average_return < -0.003:
            return "Bearish"
        return "Neutral"

    @staticmethod
    def _sector_trend(data: pd.DataFrame, sector_map: dict[str, str]) -> dict[str, str]:
        if not sector_map:
            return {}
        mapped = data.copy()
        mapped["sector"] = mapped["symbol"].map(sector_map)
        trends: dict[str, str] = {}
        for sector, group in mapped.dropna(subset=["sector"]).groupby("sector"):
            mean_return = group["return_pct"].mean()
            trends[str(sector)] = "Bullish" if mean_return > 0.003 else "Bearish" if mean_return < -0.003 else "Neutral"
        return trends

    @staticmethod
    def _health_score(data: pd.DataFrame, volatility: float) -> float:
        breadth = float((data["return_pct"] > 0).mean())
        average_return = float(data["return_pct"].mean())
        raw_score = 50 + (breadth - 0.5) * 60 + average_return * 800 - min(volatility * 250, 20)
        return round(max(0, min(100, raw_score)), 2)
