from __future__ import annotations

from datetime import UTC, datetime, timedelta
import logging
from pathlib import Path

import pandas as pd

from ai_trading_system.data_collection.providers import MarketDataProvider
from ai_trading_system.data_collection.validators import OHLCVValidator
from ai_trading_system.database.repositories import MarketDataRepository

logger = logging.getLogger(__name__)


class HistoricalDataCollector:
    """Coordinates historical OHLCV download, validation, CSV storage, and DB storage."""

    def __init__(
        self,
        provider: MarketDataProvider,
        repository: MarketDataRepository,
        validator: OHLCVValidator,
        output_dir: Path,
    ) -> None:
        self.provider = provider
        self.repository = repository
        self.validator = validator
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def collect_symbol(self, symbol: str, years: int = 1, interval: str = "1d") -> int:
        if years < 1 or years > 5:
            raise ValueError("years must be between 1 and 5.")

        end = datetime.now(UTC)
        start = end - timedelta(days=365 * years)
        raw = self.provider.get_historical_ohlcv(symbol=symbol, start=start, end=end, interval=interval)
        normalized = self._normalize(raw)
        validated = self.validator.validate(normalized)
        self._write_csv(symbol=symbol, frame=validated)
        return self.repository.upsert_historical_prices(symbol=symbol, frame=validated)

    def collect_many(self, symbols: tuple[str, ...], years: int = 1, interval: str = "1d") -> dict[str, int]:
        results: dict[str, int] = {}
        for symbol in symbols:
            try:
                results[symbol] = self.collect_symbol(symbol=symbol, years=years, interval=interval)
            except Exception:
                logger.exception("Historical collection failed", extra={"symbol": symbol})
                results[symbol] = 0
        return results

    def _write_csv(self, symbol: str, frame: pd.DataFrame) -> None:
        safe_symbol = symbol.replace(".", "_")
        frame.to_csv(self.output_dir / f"{safe_symbol}.csv", index=False)

    @staticmethod
    def _normalize(frame: pd.DataFrame) -> pd.DataFrame:
        normalized = frame.reset_index()
        normalized.columns = [str(column).lower().replace(" ", "_") for column in normalized.columns]
        rename_map = {"datetime": "date", "adj_close": "adj_close"}
        normalized = normalized.rename(columns=rename_map)
        if "date" not in normalized.columns and "index" in normalized.columns:
            normalized = normalized.rename(columns={"index": "date"})
        expected = ["date", "open", "high", "low", "close", "adj_close", "volume"]
        for column in expected:
            if column not in normalized.columns:
                normalized[column] = None
        return normalized[expected]
