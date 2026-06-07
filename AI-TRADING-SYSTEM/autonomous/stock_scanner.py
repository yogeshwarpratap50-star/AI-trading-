"""
Autonomous Stock Scanner — Phase 12.

Scans the NIFTY50 universe and ranks opportunities by a composite score built from:
  * Momentum  (price vs EMA-20, EMA-50)
  * Volume    (current vs 20-day average)
  * Sentiment (latest news sentiment score)
  * AI signal confidence

No hard-coded thresholds — everything is relative ranking.
Falls back gracefully when data is unavailable.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# Default NIFTY50 scan universe (subset; extended at runtime from settings)
_DEFAULT_SYMBOLS = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS",
    "SBIN.NS", "BHARTIARTL.NS", "ITC.NS", "LT.NS", "HINDUNILVR.NS",
    "KOTAKBANK.NS", "AXISBANK.NS", "BAJFINANCE.NS", "MARUTI.NS", "SUNPHARMA.NS",
    "TITAN.NS", "ULTRACEMCO.NS", "ASIANPAINT.NS", "WIPRO.NS", "POWERGRID.NS",
]


@dataclass
class ScanResult:
    symbol: str
    score: float                    # 0-100 composite
    momentum_score: float
    volume_score: float
    sentiment_score: float
    ai_confidence: float
    ai_action: str                  # BUY / SELL / HOLD
    current_price: float
    atr: float
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "score": round(self.score, 2),
            "momentum": round(self.momentum_score, 2),
            "volume": round(self.volume_score, 2),
            "sentiment": round(self.sentiment_score, 2),
            "ai_confidence": round(self.ai_confidence, 2),
            "ai_action": self.ai_action,
            "price": round(self.current_price, 2),
            "atr": round(self.atr, 2),
            "reasons": self.reasons,
        }


class StockScanner:
    """
    Ranks NIFTY50 stocks by opportunity score.

    Usage
    -----
    scanner = StockScanner(symbols=["RELIANCE.NS", "TCS.NS"])
    results = scanner.scan(min_score=40.0)   # returns sorted list
    """

    # Composite weights (must sum to 1.0)
    W_MOMENTUM  = 0.35
    W_VOLUME    = 0.20
    W_SENTIMENT = 0.20
    W_AI        = 0.25

    def __init__(
        self,
        symbols: list[str] | None = None,
        data_dir: str | Path = "data/historical",
        sentiment_db_path: str | Path | None = None,
        prediction_engine=None,
    ) -> None:
        self.symbols = symbols or _DEFAULT_SYMBOLS
        self.data_dir = Path(data_dir)
        self.sentiment_db_path = Path(sentiment_db_path) if sentiment_db_path else None
        self.prediction_engine = prediction_engine
        self.logger = logging.getLogger(__name__)

    def scan(self, min_score: float = 30.0, top_n: int | None = None) -> list[ScanResult]:
        """
        Scan all symbols and return BUY candidates sorted by score descending.
        """
        results = []
        for symbol in self.symbols:
            try:
                result = self._score_symbol(symbol)
                if result and result.ai_action == "BUY" and result.score >= min_score:
                    results.append(result)
            except Exception as exc:
                self.logger.debug("Scanner skip %s: %s", symbol, exc)

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_n] if top_n else results

    def scan_all(self) -> list[ScanResult]:
        """Scan all symbols regardless of action — returns full ranking."""
        results = []
        for symbol in self.symbols:
            try:
                result = self._score_symbol(symbol)
                if result:
                    results.append(result)
            except Exception as exc:
                self.logger.debug("Scanner skip %s: %s", symbol, exc)
        results.sort(key=lambda r: r.score, reverse=True)
        return results

    # ------------------------------------------------------------------
    # Per-symbol scoring
    # ------------------------------------------------------------------

    def _score_symbol(self, symbol: str) -> ScanResult | None:
        ohlcv = self._load_ohlcv(symbol)
        if ohlcv is None or len(ohlcv) < 50:
            return None

        close = ohlcv["close"].astype(float)
        volume = ohlcv["volume"].astype(float)
        high = ohlcv["high"].astype(float)
        low = ohlcv["low"].astype(float)

        # Momentum: EMA crossover + price position
        ema20 = close.ewm(span=20, adjust=False).mean()
        ema50 = close.ewm(span=50, adjust=False).mean()
        last_price = float(close.iloc[-1])
        ema20_val = float(ema20.iloc[-1])
        ema50_val = float(ema50.iloc[-1])

        # Normalise momentum into 0-100
        trend_gap = (ema20_val - ema50_val) / ema50_val * 100  # %
        mom_score = float(np.clip(50 + trend_gap * 10, 0, 100))

        # Volume: current vs 20d average
        avg_vol = float(volume.tail(20).mean())
        curr_vol = float(volume.iloc[-1])
        vol_ratio = curr_vol / avg_vol if avg_vol > 0 else 1.0
        vol_score = float(np.clip(vol_ratio * 50, 0, 100))

        # ATR (14-day)
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs(),
        ], axis=1).max(axis=1)
        atr = float(tr.rolling(14).mean().iloc[-1]) if len(tr) >= 14 else 0.0

        # Sentiment
        sent_score, sent_raw = self._get_sentiment(symbol)

        # AI prediction
        ai_action, ai_conf = self._get_ai_prediction(symbol, ohlcv)

        # AI score: confidence rescaled, penalised for SELL/HOLD
        ai_score = ai_conf if ai_action == "BUY" else max(0, 50 - ai_conf * 0.5)

        # Composite
        composite = (
            self.W_MOMENTUM  * mom_score +
            self.W_VOLUME    * vol_score +
            self.W_SENTIMENT * sent_score +
            self.W_AI        * ai_score
        )

        reasons: list[str] = []
        if trend_gap > 0:
            reasons.append(f"EMA bullish ({trend_gap:+.1f}%)")
        if vol_ratio > 1.5:
            reasons.append(f"High volume ({vol_ratio:.1f}x avg)")
        if sent_raw > 0.1:
            reasons.append(f"Positive sentiment ({sent_raw:.2f})")
        if ai_action == "BUY":
            reasons.append(f"AI BUY ({ai_conf:.0f}%)")

        return ScanResult(
            symbol=symbol,
            score=round(float(composite), 2),
            momentum_score=round(mom_score, 2),
            volume_score=round(vol_score, 2),
            sentiment_score=round(sent_score, 2),
            ai_confidence=round(ai_conf, 2),
            ai_action=ai_action,
            current_price=round(last_price, 2),
            atr=round(atr, 2),
            reasons=reasons,
        )

    # ------------------------------------------------------------------
    # Data sources
    # ------------------------------------------------------------------

    def _load_ohlcv(self, symbol: str) -> pd.DataFrame | None:
        csv_path = self.data_dir / f"{symbol.replace('.', '_')}.csv"
        if not csv_path.exists():
            return None
        df = pd.read_csv(csv_path)
        df.columns = [c.lower() for c in df.columns]
        required = {"open", "high", "low", "close", "volume"}
        if not required.issubset(df.columns):
            return None
        return df.tail(200)   # last 200 bars is enough

    def _get_sentiment(self, symbol: str) -> tuple[float, float]:
        """Returns (normalised 0-100, raw -1 to +1). Reads SQLite if available."""
        if self.sentiment_db_path and self.sentiment_db_path.exists():
            try:
                import sqlite3
                ticker = symbol.replace(".NS", "").replace(".BSE", "")
                with sqlite3.connect(str(self.sentiment_db_path), timeout=2) as conn:
                    row = conn.execute(
                        "SELECT AVG(sentiment_score) FROM sentiment_results "
                        "WHERE ticker=? ORDER BY id DESC LIMIT 10",
                        (ticker,),
                    ).fetchone()
                if row and row[0] is not None:
                    raw = float(row[0])
                    normalised = float(np.clip(50 + raw * 50, 0, 100))
                    return normalised, raw
            except Exception:
                pass
        return 50.0, 0.0   # neutral default

    def _get_ai_prediction(self, symbol: str, ohlcv: pd.DataFrame) -> tuple[str, float]:
        """Returns (action, confidence 0-100)."""
        if self.prediction_engine is None:
            return "HOLD", 0.0
        try:
            from features.feature_engineering import FeatureEngineeringService
            features, _ = FeatureEngineeringService().create_feature_dataset(ohlcv)
            result = self.prediction_engine.predict(features.tail(1))
            return result.action, float(result.confidence)
        except Exception:
            return "HOLD", 0.0
