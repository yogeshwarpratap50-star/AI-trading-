from __future__ import annotations

import csv
from datetime import UTC, datetime
import logging
from pathlib import Path

from ai_trading_system.data_collection.providers import MarketDataProvider
from ai_trading_system.database.repositories import MarketDataRepository

logger = logging.getLogger(__name__)


class LiveDataCollector:
    """Collects latest quotes and stores them in CSV and SQLite."""

    def __init__(self, provider: MarketDataProvider, repository: MarketDataRepository, output_dir: Path) -> None:
        self.provider = provider
        self.repository = repository
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def collect_symbol(self, symbol: str) -> dict[str, float | int | str | None]:
        quote = self.provider.get_live_quote(symbol)
        quote["timestamp"] = datetime.now(UTC).isoformat()
        self.repository.insert_live_quote(quote)
        self._append_csv(quote)
        logger.info("Collected live quote", extra={"symbol": symbol})
        return quote

    def _append_csv(self, quote: dict[str, float | int | str | None]) -> None:
        path = self.output_dir / "live_quotes.csv"
        write_header = not path.exists()
        with path.open("a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=["timestamp", "symbol", "last_price", "open", "day_high", "day_low", "volume"])
            if write_header:
                writer.writeheader()
            writer.writerow({key: quote.get(key) for key in writer.fieldnames})
