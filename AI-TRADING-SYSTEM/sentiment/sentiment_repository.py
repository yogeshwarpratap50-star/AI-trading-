from __future__ import annotations

from pathlib import Path
import csv

from ai_trading_system.database.connection import SQLiteConnectionManager
from sentiment.sentiment_engine import SentimentResult


class SentimentRepository:
    """Persists sentiment analysis output."""

    def __init__(
        self,
        connection_manager: SQLiteConnectionManager,
        csv_path: Path = Path("data/sentiment/sentiment.csv"),
    ) -> None:
        self.connection_manager = connection_manager
        self.csv_path = csv_path
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    def initialize(self) -> None:
        with self.connection_manager.connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS sentiment_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    headline TEXT NOT NULL,
                    ticker TEXT,
                    sentiment TEXT NOT NULL,
                    sentiment_score REAL NOT NULL,
                    confidence_score REAL NOT NULL,
                    source_url TEXT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(headline, ticker, source_url)
                )
                """
            )

    def save_many(self, results: list[SentimentResult]) -> int:
        inserted = 0
        with self.connection_manager.connect() as connection:
            for result in results:
                cursor = connection.execute(
                    """
                    INSERT OR IGNORE INTO sentiment_results
                        (headline, ticker, sentiment, sentiment_score, confidence_score, source_url)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        result.headline,
                        result.ticker,
                        result.sentiment,
                        result.sentiment_score,
                        result.confidence_score,
                        result.source_url,
                    ),
                )
                inserted += cursor.rowcount
        self._append_csv(results)
        return inserted

    def latest(self, limit: int = 100) -> list[dict]:
        with self.connection_manager.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM sentiment_results ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def _append_csv(self, results: list[SentimentResult]) -> None:
        if not results:
            return
        fieldnames = ["headline", "ticker", "sentiment", "sentiment_score", "confidence_score", "source_url"]
        write_header = not self.csv_path.exists()
        with self.csv_path.open("a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            for result in results:
                writer.writerow(result.__dict__)
