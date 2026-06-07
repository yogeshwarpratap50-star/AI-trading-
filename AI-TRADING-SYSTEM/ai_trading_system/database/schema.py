from __future__ import annotations

import logging

from ai_trading_system.database.connection import SQLiteConnectionManager

logger = logging.getLogger(__name__)


class DatabaseInitializer:
    """Creates database schema required by Phase 1."""

    def __init__(self, connection_manager: SQLiteConnectionManager) -> None:
        self.connection_manager = connection_manager

    def initialize(self) -> None:
        with self.connection_manager.connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS historical_prices (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    date TEXT NOT NULL,
                    open REAL NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    close REAL NOT NULL,
                    adj_close REAL,
                    volume INTEGER NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(symbol, date)
                );

                CREATE TABLE IF NOT EXISTS live_quotes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timestamp TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    last_price REAL,
                    open REAL,
                    day_high REAL,
                    day_low REAL,
                    volume INTEGER
                );

                CREATE TABLE IF NOT EXISTS data_collection_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    started_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    completed_at TEXT,
                    message TEXT
                );
                """
            )
        logger.info("Database schema initialized")
