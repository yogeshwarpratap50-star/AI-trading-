from __future__ import annotations

import logging

import pandas as pd

from ai_trading_system.database.connection import SQLiteConnectionManager

logger = logging.getLogger(__name__)


class MarketDataRepository:
    """Persists market data to SQLite."""

    def __init__(self, connection_manager: SQLiteConnectionManager) -> None:
        self.connection_manager = connection_manager

    def upsert_historical_prices(self, symbol: str, frame: pd.DataFrame) -> int:
        rows = []
        for record in frame.to_dict(orient="records"):
            rows.append(
                (
                    symbol,
                    record["date"].isoformat(),
                    float(record["open"]),
                    float(record["high"]),
                    float(record["low"]),
                    float(record["close"]),
                    float(record["adj_close"]) if pd.notna(record.get("adj_close")) else None,
                    int(record["volume"]),
                )
            )

        with self.connection_manager.connect() as connection:
            connection.executemany(
                """
                INSERT INTO historical_prices
                    (symbol, date, open, high, low, close, adj_close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(symbol, date) DO UPDATE SET
                    open=excluded.open,
                    high=excluded.high,
                    low=excluded.low,
                    close=excluded.close,
                    adj_close=excluded.adj_close,
                    volume=excluded.volume
                """,
                rows,
            )
        logger.info("Persisted historical prices", extra={"symbol": symbol, "rows": len(rows)})
        return len(rows)

    def insert_live_quote(self, quote: dict[str, float | int | str | None]) -> None:
        with self.connection_manager.connect() as connection:
            connection.execute(
                """
                INSERT INTO live_quotes (symbol, last_price, open, day_high, day_low, volume)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    quote["symbol"],
                    quote.get("last_price"),
                    quote.get("open"),
                    quote.get("day_high"),
                    quote.get("day_low"),
                    quote.get("volume"),
                ),
            )
