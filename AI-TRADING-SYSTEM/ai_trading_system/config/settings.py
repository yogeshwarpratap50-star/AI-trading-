from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings loaded from environment variables and .env files."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
    )

    app_name: str = "AI-TRADING-SYSTEM"
    environment: Literal["local", "dev", "staging", "prod"] = "local"
    log_level: str = "INFO"

    database_url: str = "sqlite:///database/trading.db"
    historical_data_dir: Path = Path("data/historical")
    live_data_dir: Path = Path("data/live")
    nifty50_symbols: tuple[str, ...] = Field(
        default=(
            "RELIANCE.NS",
            "TCS.NS",
            "HDFCBANK.NS",
            "ICICIBANK.NS",
            "INFY.NS",
            "SBIN.NS",
            "BHARTIARTL.NS",
            "ITC.NS",
            "LT.NS",
            "HINDUNILVR.NS",
        )
    )
    default_history_years: int = Field(default=1, ge=1, le=5)
    request_timeout_seconds: int = Field(default=30, ge=1, le=300)
    retry_attempts: int = Field(default=3, ge=1, le=10)
    retry_wait_seconds: int = Field(default=2, ge=1, le=60)

    alpha_vantage_api_key: str | None = None
    marketaux_api_key: str | None = None
    dashboard_refresh_seconds: int = Field(default=30, ge=10, le=300)
    rss_feed_urls: tuple[str, ...] = Field(
        default=(
            "https://www.moneycontrol.com/rss/business.xml",
            "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
        )
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
