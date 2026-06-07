from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
import logging

import pandas as pd
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed
import yfinance as yf

from ai_trading_system.data_collection.exceptions import DataCollectionError

logger = logging.getLogger(__name__)


class MarketDataProvider(ABC):
    """Abstract market data provider contract."""

    @abstractmethod
    def get_historical_ohlcv(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """Return historical OHLCV bars for a symbol."""

    @abstractmethod
    def get_live_quote(self, symbol: str) -> dict[str, float | int | str | None]:
        """Return the latest available quote for a symbol."""


class YahooFinanceProvider(MarketDataProvider):
    """Yahoo Finance implementation suitable for research and paper workflows."""

    def __init__(self, retry_attempts: int = 3, retry_wait_seconds: int = 2) -> None:
        self.retry_attempts = retry_attempts
        self.retry_wait_seconds = retry_wait_seconds

    def get_historical_ohlcv(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        interval: str = "1d",
    ) -> pd.DataFrame:
        return self._download_history(symbol=symbol, start=start, end=end, interval=interval)

    def get_live_quote(self, symbol: str) -> dict[str, float | int | str | None]:
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.fast_info
            return {
                "symbol": symbol,
                "last_price": float(info.get("last_price")) if info.get("last_price") is not None else None,
                "open": float(info.get("open")) if info.get("open") is not None else None,
                "day_high": float(info.get("day_high")) if info.get("day_high") is not None else None,
                "day_low": float(info.get("day_low")) if info.get("day_low") is not None else None,
                "volume": int(info.get("last_volume")) if info.get("last_volume") is not None else None,
            }
        except Exception as exc:
            raise DataCollectionError(f"Failed to fetch live quote for {symbol}.") from exc

    def _download_history(self, symbol: str, start: datetime, end: datetime, interval: str) -> pd.DataFrame:
        @retry(
            stop=stop_after_attempt(self.retry_attempts),
            wait=wait_fixed(self.retry_wait_seconds),
            retry=retry_if_exception_type(Exception),
            reraise=True,
        )
        def _download() -> pd.DataFrame:
            logger.info("Downloading historical data", extra={"symbol": symbol, "interval": interval})
            data = yf.download(
                tickers=symbol,
                start=start,
                end=end,
                interval=interval,
                auto_adjust=False,
                progress=False,
                threads=False,
            )
            if data.empty:
                raise DataCollectionError(f"No historical data returned for {symbol}.")
            return data

        try:
            return _download()
        except Exception as exc:
            raise DataCollectionError(f"Failed to download historical data for {symbol}.") from exc
