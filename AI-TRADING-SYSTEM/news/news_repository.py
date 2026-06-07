from __future__ import annotations

import csv
from pathlib import Path

from ai_trading_system.database.connection import SQLiteConnectionManager
from news.news_collector import NewsArticle


class NewsRepository:
    """Persists financial news to SQLite and CSV."""

    def __init__(self, connection_manager: SQLiteConnectionManager, csv_path: Path = Path("data/news/news.csv")) -> None:
        self.connection_manager = connection_manager
        self.csv_path = csv_path
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    def initialize(self) -> None:
        with self.connection_manager.connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS news_articles (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    headline TEXT NOT NULL,
                    summary TEXT,
                    source TEXT,
                    url TEXT,
                    timestamp TEXT NOT NULL,
                    ticker TEXT,
                    category TEXT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(url, headline, ticker)
                )
                """
            )

    def save_many(self, articles: list[NewsArticle]) -> int:
        inserted = 0
        with self.connection_manager.connect() as connection:
            for article in articles:
                cursor = connection.execute(
                    """
                    INSERT OR IGNORE INTO news_articles
                        (headline, summary, source, url, timestamp, ticker, category)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        article.headline,
                        article.summary,
                        article.source,
                        article.url,
                        article.timestamp.isoformat(),
                        article.ticker,
                        article.category,
                    ),
                )
                inserted += cursor.rowcount
        self._append_csv(articles)
        return inserted

    def latest(self, limit: int = 100) -> list[dict]:
        with self.connection_manager.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM news_articles ORDER BY timestamp DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def _append_csv(self, articles: list[NewsArticle]) -> None:
        if not articles:
            return
        fieldnames = ["headline", "summary", "source", "url", "timestamp", "ticker", "category"]
        write_header = not self.csv_path.exists()
        with self.csv_path.open("a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            for article in articles:
                writer.writerow(article.to_record())
