from datetime import UTC, datetime

from ai_trading_system.database.connection import SQLiteConnectionManager
from news.news_collector import NewsArticle
from sentiment.sentiment_engine import SentimentEngine
from sentiment.sentiment_repository import SentimentRepository


def positive_article() -> NewsArticle:
    return NewsArticle(
        headline="Infosys shares rally after strong profit beat",
        summary="Analysts upgrade the stock after positive growth.",
        source="Unit Test",
        url="https://example.com/infosys",
        timestamp=datetime(2025, 1, 1, tzinfo=UTC),
        ticker="INFY.NS",
        category="earnings",
    )


def test_sentiment_engine_classifies_positive_news() -> None:
    result = SentimentEngine().analyze_article(positive_article())

    assert result.sentiment == "Positive"
    assert result.sentiment_score > 0
    assert result.confidence_score > 0


def test_sentiment_repository_persists_results(tmp_path) -> None:
    manager = SQLiteConnectionManager(f"sqlite:///{tmp_path / 'sentiment.db'}")
    repository = SentimentRepository(manager, tmp_path / "sentiment.csv")
    result = SentimentEngine().analyze_article(positive_article())

    assert repository.save_many([result]) == 1
    assert repository.save_many([result]) == 0
    assert repository.latest()[0]["sentiment"] == "Positive"
