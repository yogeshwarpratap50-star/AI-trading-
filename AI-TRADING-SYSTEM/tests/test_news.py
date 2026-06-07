from datetime import UTC, datetime

from ai_trading_system.database.connection import SQLiteConnectionManager
from news.news_collector import NewsArticle, NewsCollector
from news.news_repository import NewsRepository


class FakeNewsProvider:
    def fetch(self, tickers: list[str], limit: int = 25) -> list[NewsArticle]:
        return [
            NewsArticle(
                headline="Reliance reports strong profit growth",
                summary="Shares may gain after strong quarterly numbers.",
                source="Unit Test",
                url="https://example.com/reliance",
                timestamp=datetime(2025, 1, 1, tzinfo=UTC),
                ticker=tickers[0],
                category="earnings",
            ),
            NewsArticle(
                headline="Reliance reports strong profit growth",
                summary="Duplicate",
                source="Unit Test",
                url="https://example.com/reliance",
                timestamp=datetime(2025, 1, 1, tzinfo=UTC),
                ticker=tickers[0],
                category="earnings",
            ),
        ]


def test_news_collector_deduplicates_articles() -> None:
    articles = NewsCollector([FakeNewsProvider()]).collect(["RELIANCE.NS"])

    assert len(articles) == 1
    assert articles[0].ticker == "RELIANCE.NS"


def test_news_repository_persists_unique_articles(tmp_path) -> None:
    manager = SQLiteConnectionManager(f"sqlite:///{tmp_path / 'news.db'}")
    repository = NewsRepository(manager, tmp_path / "news.csv")
    articles = NewsCollector([FakeNewsProvider()]).collect(["RELIANCE.NS"])

    inserted = repository.save_many(articles)
    inserted_again = repository.save_many(articles)

    assert inserted == 1
    assert inserted_again == 0
    assert len(repository.latest()) == 1
    assert (tmp_path / "news.csv").exists()
