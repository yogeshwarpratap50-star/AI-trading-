from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import logging
from typing import Protocol
from urllib.parse import urlencode

import requests
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed


@dataclass(frozen=True)
class NewsArticle:
    headline: str
    summary: str
    source: str
    url: str
    timestamp: datetime
    ticker: str
    category: str

    def to_record(self) -> dict[str, str]:
        record = asdict(self)
        record["timestamp"] = self.timestamp.isoformat()
        return record


class NewsProvider(Protocol):
    def fetch(self, tickers: list[str], limit: int = 25) -> list[NewsArticle]:
        """Fetch news articles for tickers."""


class AlphaVantageNewsProvider:
    def __init__(self, api_key: str, timeout_seconds: int = 30) -> None:
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.logger = logging.getLogger(self.__class__.__name__)

    def fetch(self, tickers: list[str], limit: int = 25) -> list[NewsArticle]:
        if not self.api_key:
            return []
        params = urlencode(
            {
                "function": "NEWS_SENTIMENT",
                "tickers": ",".join(tickers),
                "limit": limit,
                "apikey": self.api_key,
            }
        )
        payload = self._get_json(f"https://www.alphavantage.co/query?{params}")
        articles: list[NewsArticle] = []
        for item in payload.get("feed", []):
            parsed_time = self._parse_alpha_time(item.get("time_published"))
            ticker = self._first_matching_ticker(item.get("ticker_sentiment", []), tickers)
            articles.append(
                NewsArticle(
                    headline=item.get("title", ""),
                    summary=item.get("summary", ""),
                    source=item.get("source", "Alpha Vantage"),
                    url=item.get("url", ""),
                    timestamp=parsed_time,
                    ticker=ticker,
                    category="market_news",
                )
            )
        return articles

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2), retry=retry_if_exception_type(requests.RequestException), reraise=True)
    def _get_json(self, url: str) -> dict:
        response = requests.get(url, timeout=self.timeout_seconds)
        response.raise_for_status()
        return response.json()

    @staticmethod
    def _parse_alpha_time(value: str | None) -> datetime:
        if not value:
            return datetime.now(UTC)
        return datetime.strptime(value, "%Y%m%dT%H%M%S").replace(tzinfo=UTC)

    @staticmethod
    def _first_matching_ticker(items: list[dict], tickers: list[str]) -> str:
        candidates = {ticker.upper() for ticker in tickers}
        for item in items:
            ticker = str(item.get("ticker", "")).upper()
            if ticker in candidates:
                return ticker
        return tickers[0] if tickers else ""


class MarketauxNewsProvider:
    def __init__(self, api_key: str, timeout_seconds: int = 30) -> None:
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds

    def fetch(self, tickers: list[str], limit: int = 25) -> list[NewsArticle]:
        if not self.api_key:
            return []
        params = urlencode({"symbols": ",".join(tickers), "limit": limit, "api_token": self.api_key})
        payload = self._get_json(f"https://api.marketaux.com/v1/news/all?{params}")
        articles: list[NewsArticle] = []
        for item in payload.get("data", []):
            timestamp = datetime.fromisoformat(item.get("published_at", "").replace("Z", "+00:00"))
            entities = item.get("entities", [])
            ticker = entities[0].get("symbol", tickers[0] if tickers else "") if entities else (tickers[0] if tickers else "")
            articles.append(
                NewsArticle(
                    headline=item.get("title", ""),
                    summary=item.get("description", ""),
                    source=item.get("source", "Marketaux"),
                    url=item.get("url", ""),
                    timestamp=timestamp,
                    ticker=ticker,
                    category=item.get("snippet", "market_news")[:80],
                )
            )
        return articles

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2), retry=retry_if_exception_type(requests.RequestException), reraise=True)
    def _get_json(self, url: str) -> dict:
        response = requests.get(url, timeout=self.timeout_seconds)
        response.raise_for_status()
        return response.json()


class YahooFinanceNewsProvider:
    def fetch(self, tickers: list[str], limit: int = 25) -> list[NewsArticle]:
        import yfinance as yf

        articles: list[NewsArticle] = []
        per_ticker_limit = max(1, limit // max(len(tickers), 1))
        for ticker in tickers:
            for item in yf.Ticker(ticker).news[:per_ticker_limit]:
                content = item.get("content", item)
                published = content.get("pubDate") or content.get("providerPublishTime")
                timestamp = self._parse_timestamp(published)
                articles.append(
                    NewsArticle(
                        headline=content.get("title", ""),
                        summary=content.get("summary", ""),
                        source=content.get("provider", {}).get("displayName", "Yahoo Finance"),
                        url=content.get("canonicalUrl", {}).get("url", content.get("link", "")),
                        timestamp=timestamp,
                        ticker=ticker,
                        category=content.get("contentType", "market_news"),
                    )
                )
        return articles

    @staticmethod
    def _parse_timestamp(value: object) -> datetime:
        if isinstance(value, int):
            return datetime.fromtimestamp(value, tz=UTC)
        if isinstance(value, str):
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        return datetime.now(UTC)


class RSSNewsProvider:
    def __init__(self, feed_urls: list[str], timeout_seconds: int = 30) -> None:
        self.feed_urls = feed_urls
        self.timeout_seconds = timeout_seconds

    def fetch(self, tickers: list[str], limit: int = 25) -> list[NewsArticle]:
        import feedparser

        articles: list[NewsArticle] = []
        for feed_url in self.feed_urls:
            response = requests.get(feed_url, timeout=self.timeout_seconds)
            response.raise_for_status()
            feed = feedparser.parse(response.text)
            for entry in feed.entries[:limit]:
                ticker = self._infer_ticker(entry.get("title", ""), tickers)
                articles.append(
                    NewsArticle(
                        headline=entry.get("title", ""),
                        summary=entry.get("summary", ""),
                        source=feed.feed.get("title", "RSS"),
                        url=entry.get("link", ""),
                        timestamp=self._parse_timestamp(entry.get("published")),
                        ticker=ticker,
                        category="rss",
                    )
                )
        return articles[:limit]

    @staticmethod
    def _infer_ticker(text: str, tickers: list[str]) -> str:
        upper = text.upper()
        for ticker in tickers:
            if ticker.upper().replace(".NS", "") in upper:
                return ticker
        return tickers[0] if tickers else ""

    @staticmethod
    def _parse_timestamp(value: str | None) -> datetime:
        if not value:
            return datetime.now(UTC)
        try:
            return datetime(*__import__("email.utils").utils.parsedate(value)[:6], tzinfo=UTC)
        except Exception:
            return datetime.now(UTC)


class NewsCollector:
    """Coordinates news collection across multiple providers."""

    def __init__(self, providers: list[NewsProvider]) -> None:
        self.providers = providers
        self.logger = logging.getLogger(self.__class__.__name__)

    def collect(self, tickers: list[str], limit: int = 25) -> list[NewsArticle]:
        articles: list[NewsArticle] = []
        for provider in self.providers:
            try:
                articles.extend(provider.fetch(tickers, limit=limit))
            except Exception:
                self.logger.exception("News provider failed", extra={"provider": provider.__class__.__name__})
        deduped = self._deduplicate(articles)
        self.logger.info("News collected", extra={"articles": len(deduped)})
        return deduped

    @staticmethod
    def _deduplicate(articles: list[NewsArticle]) -> list[NewsArticle]:
        seen: set[str] = set()
        deduped: list[NewsArticle] = []
        for article in articles:
            key = article.url or f"{article.headline}|{article.ticker}|{article.timestamp.date()}"
            if key in seen:
                continue
            seen.add(key)
            deduped.append(article)
        return deduped
