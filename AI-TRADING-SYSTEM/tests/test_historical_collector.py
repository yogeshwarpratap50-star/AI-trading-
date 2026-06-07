from datetime import datetime

import pandas as pd

from ai_trading_system.data_collection.historical import HistoricalDataCollector
from ai_trading_system.data_collection.providers import MarketDataProvider
from ai_trading_system.data_collection.validators import OHLCVValidator


class FakeProvider(MarketDataProvider):
    def get_historical_ohlcv(self, symbol: str, start: datetime, end: datetime, interval: str = "1d") -> pd.DataFrame:
        return pd.DataFrame(
            {
                "Date": ["2025-01-01"],
                "Open": [100],
                "High": [110],
                "Low": [95],
                "Close": [105],
                "Adj Close": [105],
                "Volume": [1000],
            }
        ).set_index("Date")

    def get_live_quote(self, symbol: str) -> dict[str, float | int | str | None]:
        return {"symbol": symbol, "last_price": 100.0}


class FakeRepository:
    def upsert_historical_prices(self, symbol: str, frame: pd.DataFrame) -> int:
        self.symbol = symbol
        self.frame = frame
        return len(frame)


def test_historical_collector_writes_csv_and_repository(tmp_path) -> None:
    repository = FakeRepository()
    collector = HistoricalDataCollector(FakeProvider(), repository, OHLCVValidator(), tmp_path)

    rows = collector.collect_symbol("RELIANCE.NS", years=1)

    assert rows == 1
    assert repository.symbol == "RELIANCE.NS"
    assert (tmp_path / "RELIANCE_NS.csv").exists()
