from __future__ import annotations

from dataclasses import dataclass
import logging
import math
import re

from news.news_collector import NewsArticle


POSITIVE_WORDS = {
    "beat",
    "bullish",
    "buy",
    "gain",
    "gains",
    "growth",
    "higher",
    "outperform",
    "positive",
    "profit",
    "rally",
    "record",
    "strong",
    "surge",
    "upgrade",
}

NEGATIVE_WORDS = {
    "bearish",
    "concern",
    "cut",
    "decline",
    "downgrade",
    "fall",
    "falls",
    "loss",
    "lower",
    "miss",
    "negative",
    "pressure",
    "risk",
    "sell",
    "weak",
}


@dataclass(frozen=True)
class SentimentResult:
    headline: str
    ticker: str
    sentiment: str
    sentiment_score: float
    confidence_score: float
    source_url: str


class SentimentEngine:
    """Rule-based sentiment engine for financial headlines and summaries."""

    def __init__(self) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)

    def analyze_article(self, article: NewsArticle) -> SentimentResult:
        text = f"{article.headline} {article.summary}"
        tokens = re.findall(r"[A-Za-z]+", text.lower())
        positive = sum(token in POSITIVE_WORDS for token in tokens)
        negative = sum(token in NEGATIVE_WORDS for token in tokens)
        raw_score = positive - negative
        sentiment_score = math.tanh(raw_score / 3)
        confidence_score = min(1.0, abs(raw_score) / max(len(tokens), 1) * 8)
        sentiment = "Neutral"
        if sentiment_score > 0.1:
            sentiment = "Positive"
        elif sentiment_score < -0.1:
            sentiment = "Negative"
        result = SentimentResult(
            headline=article.headline,
            ticker=article.ticker,
            sentiment=sentiment,
            sentiment_score=round(sentiment_score, 4),
            confidence_score=round(confidence_score, 4),
            source_url=article.url,
        )
        self.logger.info("Article sentiment analyzed", extra={"ticker": article.ticker, "sentiment": sentiment})
        return result

    def analyze_many(self, articles: list[NewsArticle]) -> list[SentimentResult]:
        return [self.analyze_article(article) for article in articles]
